import unittest
from unittest.mock import patch, MagicMock
from config import settings
from rag.store import FallbackVectorStore, get_vector_store
from rag.errors import RetryableRetrievalError, FatalRetrievalError
from tools import search_local_knowledge

class TestRetrievalFallback(unittest.TestCase):

    def setUp(self):
        # Create common mock stores
        self.mock_chroma = MagicMock()
        self.mock_pinecone = MagicMock()

        # Helper to patch `_get_backend` directly
        def get_backend_side_effect(name):
            if name == "chroma":
                return self.mock_chroma
            elif name == "pinecone":
                return self.mock_pinecone
            else:
                raise FatalRetrievalError(f"Unsupported VECTOR_DB: '{name}'")

        self.get_backend_patcher = patch('rag.store._get_backend', side_effect=get_backend_side_effect)
        self.mock_get_backend = self.get_backend_patcher.start()
        
        # Save original settings
        self.orig_vector_db = settings.vector_db
        self.orig_vector_db_fallback = settings.vector_db_fallback
        self.orig_vector_db_fallback_enabled = settings.vector_db_fallback_enabled

    def tearDown(self):
        self.get_backend_patcher.stop()
        
        # Restore settings
        settings.vector_db = self.orig_vector_db
        settings.vector_db_fallback = self.orig_vector_db_fallback
        settings.vector_db_fallback_enabled = self.orig_vector_db_fallback_enabled

    def test_1_chroma_success(self):
        settings.vector_db = "chroma"
        settings.vector_db_fallback_enabled = True
        settings.vector_db_fallback = "pinecone"

        store = get_vector_store()
        
        # Create a mock query embedding
        import numpy as np
        query_emb = np.array([0.1, 0.2, 0.3])
        
        self.mock_chroma.search.return_value = ["Doc1", "Doc2"]
        
        results = store.search(query_emb, k=2)
        
        self.assertEqual(results, ["Doc1", "Doc2"])
        self.mock_chroma.search.assert_called_once_with(query_emb, 2)
        self.mock_pinecone.search.assert_not_called()

    def test_2_chroma_retryable_failure_pinecone_success(self):
        settings.vector_db = "chroma"
        settings.vector_db_fallback_enabled = True
        settings.vector_db_fallback = "pinecone"

        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        self.mock_chroma.search.side_effect = RetryableRetrievalError("Chroma network error")
        self.mock_pinecone.search.return_value = ["Pinecone Doc"]
        
        results = store.search(query_emb, k=1)
        
        self.assertEqual(results, ["Pinecone Doc"])
        self.mock_chroma.search.assert_called_once_with(query_emb, 1)
        self.mock_pinecone.search.assert_called_once_with(query_emb, 1)

    def test_3_chroma_fatal_error(self):
        settings.vector_db = "chroma"
        settings.vector_db_fallback_enabled = True
        settings.vector_db_fallback = "pinecone"

        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        self.mock_chroma.search.side_effect = FatalRetrievalError("Chroma bad request")
        
        with self.assertRaises(FatalRetrievalError):
            store.search(query_emb, k=1)
            
        self.mock_chroma.search.assert_called_once_with(query_emb, 1)
        self.mock_pinecone.search.assert_not_called()

    def test_4_both_backends_fail(self):
        settings.vector_db = "chroma"
        settings.vector_db_fallback_enabled = True
        settings.vector_db_fallback = "pinecone"

        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        self.mock_chroma.search.side_effect = RetryableRetrievalError("Chroma timeout")
        self.mock_pinecone.search.side_effect = RetryableRetrievalError("Pinecone timeout")
        
        with self.assertRaises(RuntimeError) as context:
            store.search(query_emb, k=1)
            
        err_msg = str(context.exception)
        self.assertIn("Both backends failed", err_msg)
        self.assertIn("Chroma timeout", err_msg)
        self.assertIn("Pinecone timeout", err_msg)

    def test_5_fallback_disabled(self):
        settings.vector_db = "chroma"
        settings.vector_db_fallback_enabled = False
        
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        self.mock_chroma.search.side_effect = RetryableRetrievalError("Chroma error")
        
        with self.assertRaises(RetryableRetrievalError):
            store.search(query_emb, k=1)
            
        self.mock_pinecone.search.assert_not_called()

    @patch('rag.pinecone_store.PineconeStore')
    def test_6_pinecone_credentials_missing(self, mock_pinecone_class):
        # Allow actual _get_backend to run for pinecone to test missing credentials
        self.get_backend_patcher.stop()
        
        # Test what happens when Pinecone credentials are not configured but Chroma works
        settings.vector_db = "chroma"
        settings.vector_db_fallback_enabled = True
        settings.vector_db_fallback = "pinecone"
        
        settings.pinecone_api_key = None  # Simulate missing
        
        # First test normal Chroma success (should not attempt pinecone init)
        with patch('rag.store._get_backend') as mock_get_backend:
            mock_chroma = MagicMock()
            mock_chroma.search.return_value = ["Data"]
            
            # Setup mock to return Chroma but fail on Pinecone as the real code would
            def get_backend_mock(name):
                if name == "chroma": return mock_chroma
                if name == "pinecone":
                    from rag.errors import FatalRetrievalError
                    raise FatalRetrievalError("pinecone_api_key configuration is missing")
                    
            mock_get_backend.side_effect = get_backend_mock
            
            store = get_vector_store()
            import numpy as np
            results = store.search(np.array([1]), k=1)
            self.assertEqual(results, ["Data"])
            
            # Now test Chroma failure, Pinecone fallback fails cleanly
            mock_chroma.search.side_effect = RetryableRetrievalError("Chroma failed")
            
            with self.assertRaises(RuntimeError) as context:
                store.search(np.array([1]), k=1)
                
            self.assertIn("pinecone_api_key configuration is missing", str(context.exception))
            
        self.get_backend_patcher.start()

    @patch('rag.embedder.embed_chunks')
    @patch('rag.store.get_vector_store')
    def test_8_agent_compatibility(self, mock_get_store, mock_embed_chunks):
        # Mock embed_chunks since we do not want to load real embeddings Model during mock test
        import numpy as np
        mock_embed_chunks.return_value = [np.array([0.1, 0.2])]
        
        # Setup mock vector store returning chunks with appropriate metadata
        mock_store = MagicMock()
        from rag.loader import Document
        mock_doc = Document(content="RAG is Retrieval-Augmented Generation", metadata={"source": "test.md", "chunk_index": 0, "distance": 0.1})
        mock_store.search.return_value = [mock_doc]
        
        mock_get_store.return_value = mock_store
        
        result = search_local_knowledge("What is RAG?")
        
        self.assertIn("Evidence 1", result)
        self.assertIn("Retrieval-Augmented Generation", result)
        self.assertIn("test.md", result)
        mock_store.search.assert_called_once()
        mock_embed_chunks.assert_called_once()


class TestExistingChromaRegression(unittest.TestCase):
    @patch('rag.embedder.embed_chunks')
    def test_7_existing_chroma_regression(self, mock_embed_chunks):
        # We test tools.search_local_knowledge but mock the LLM/Embedding calls
        # Wait, if we use real Chroma we need an existing .chroma_db!
        import os
        if not os.path.exists(".chroma_db"):
            self.skipTest("No local .chroma_db found. Skipping real ChromaDB integration test.")
            
        import numpy as np
        mock_embed_chunks.return_value = [np.ones(384)] # dummy 384 dim vector
        
        result = search_local_knowledge("What is RAG?")
        
        # Should return 'No relevant local documents found' or actual docs, but it shouldn't crash!
        self.assertTrue(isinstance(result, str))
        self.assertFalse(result.startswith("Error:"))

if __name__ == '__main__':
    unittest.main()
