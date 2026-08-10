import unittest
from unittest.mock import patch, MagicMock
from config import settings
from rag.store import FusionVectorStore, get_vector_store
from rag.loader import Document

class TestRetrievalFusion(unittest.TestCase):
    def setUp(self):
        self.mock_chroma = MagicMock()
        self.mock_pinecone = MagicMock()

        def get_backend_side_effect(name):
            if name == "chroma":
                return self.mock_chroma
            elif name == "pinecone":
                return self.mock_pinecone
            else:
                raise ValueError(f"Unknown: {name}")

        self.get_backend_patcher = patch('rag.store._get_backend', side_effect=get_backend_side_effect)
        self.mock_get_backend = self.get_backend_patcher.start()
        
        self.orig_fusion = getattr(settings, 'retrieval_fusion_enabled', False)
        settings.retrieval_fusion_enabled = True

    def tearDown(self):
        self.get_backend_patcher.stop()
        settings.retrieval_fusion_enabled = self.orig_fusion

    def test_2_rrf_fusion(self):
        settings.retrieval_final_k = 4
        store = get_vector_store()
        self.assertIsInstance(store, FusionVectorStore)
        
        import numpy as np
        query_emb = np.array([0.1])
        
        docA = Document(content="A", metadata={"source": "A.txt", "chunk_index": 0})
        docB = Document(content="B", metadata={"source": "B.txt", "chunk_index": 0})
        docC = Document(content="C", metadata={"source": "C.txt", "chunk_index": 0})
        docD = Document(content="D", metadata={"source": "D.txt", "chunk_index": 0})
        
        # Chroma ranks: A (1), B (2), C (3)
        self.mock_chroma.search.return_value = [docA, docB, docC]
        # Pinecone ranks: B (1), A (2), D (3)
        self.mock_pinecone.search.return_value = [docB, docA, docD]
        
        # Expected scores:
        # A: 1/61 + 1/62 = 0.01639 + 0.01612 = 0.03251
        # B: 1/62 + 1/61 = 0.01612 + 0.01639 = 0.03251
        # (A and B are tied, but both are much higher than C or D)
        # We will retrieve top 3, so D will be dropped, or C. Both C and D have the same rank (3) but only in one backend.
        # Actually let's just assert final length is 4 and A, B are in it.
        
        results = store.search(query_emb, k=4)
        
        self.assertEqual(len(results), 4)
        # Top 2 should be A and B in any order
        sources = [d.metadata["source"] for d in results[:2]]
        self.assertIn("A.txt", sources)
        self.assertIn("B.txt", sources)
        
        # Check backend metadata
        for r in results:
            if r.metadata["source"] in ["A.txt", "B.txt"]:
                self.assertEqual(r.metadata["backend"], "chroma+pinecone")
            elif r.metadata["source"] == "C.txt":
                self.assertEqual(r.metadata["backend"], "chroma")
            elif r.metadata["source"] == "D.txt":
                self.assertEqual(r.metadata["backend"], "pinecone")

    def test_3_deduplication(self):
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        doc = Document(content="Same chunk", metadata={"source": "rag.txt", "chunk_index": 0})
        
        self.mock_chroma.search.return_value = [doc]
        self.mock_pinecone.search.return_value = [doc]
        
        results = store.search(query_emb, k=3)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Same chunk")
        self.assertEqual(results[0].metadata["backend"], "chroma+pinecone")

    def test_5_final_k(self):
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        docs = [Document(content=str(i), metadata={"source": f"{i}.txt", "chunk_index": 0}) for i in range(10)]
        self.mock_chroma.search.return_value = docs
        self.mock_pinecone.search.return_value = []
        
        results = store.search(query_emb, k=3)
        self.assertEqual(len(results), 3)

    def test_6_fewer_results(self):
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        docs = [Document(content=str(i), metadata={"source": f"{i}.txt", "chunk_index": 0}) for i in range(2)]
        self.mock_chroma.search.return_value = docs
        self.mock_pinecone.search.return_value = []
        
        results = store.search(query_emb, k=5)
        self.assertEqual(len(results), 2)

    def test_7_empty_results(self):
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        self.mock_chroma.search.return_value = []
        self.mock_pinecone.search.return_value = []
        
        results = store.search(query_emb, k=5)
        self.assertEqual(results, [])

    def test_8_pinecone_unavailable(self):
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        doc = Document(content="Chroma ok", metadata={"source": "rag.txt", "chunk_index": 0})
        self.mock_chroma.search.return_value = [doc]
        self.mock_pinecone.search.side_effect = RuntimeError("Pinecone down")
        
        results = store.search(query_emb, k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Chroma ok")

    def test_9_fusion_disabled(self):
        settings.retrieval_fusion_enabled = False
        store = get_vector_store()
        # Should return FallbackVectorStore, not FusionVectorStore
        from rag.store import FallbackVectorStore
        self.assertIsInstance(store, FallbackVectorStore)

    def test_12_no_raw_score_comparison(self):
        # We ensure fusion happens by rank (normalized_score calculation logic)
        store = get_vector_store()
        import numpy as np
        query_emb = np.array([0.1])
        
        doc1 = Document(content="Chroma high dist (bad)", metadata={"source": "1.txt", "chunk_index": 0, "distance": 0.99})
        doc2 = Document(content="Pinecone low score (bad)", metadata={"source": "2.txt", "chunk_index": 0, "distance": 0.01})
        
        self.mock_chroma.search.return_value = [doc1]
        self.mock_pinecone.search.return_value = [doc2]
        
        results = store.search(query_emb, k=3)
        self.assertEqual(len(results), 2)
        # Ranks will be equal, so arbitrary order, but both are present without throwing exceptions about float comparisons
        # Also raw score is retained
        for r in results:
            self.assertIn("raw_score", r.metadata)


if __name__ == '__main__':
    unittest.main()
