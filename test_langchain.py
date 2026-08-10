import unittest
from unittest.mock import patch, MagicMock

from langchain_integration import (
    to_langchain_document,
    get_rag_prompt_template,
    get_reflection_prompt_template,
    LangChainRetrieverAdapter
)
from rag.loader import Document as CustomDocument

class TestLangChainIntegration(unittest.TestCase):
    
    def test_1_document_conversion(self):
        # TEST 1: LangChain Document conversion preserves page_content, source, chunk_index, retrieval metadata
        custom_doc = CustomDocument(
            content="This is a test chunk.",
            metadata={
                "source": "test.txt",
                "chunk_index": 2,
                "distance": 0.85
            }
        )
        
        lc_doc = to_langchain_document(custom_doc)
        self.assertEqual(lc_doc.page_content, "This is a test chunk.")
        self.assertEqual(lc_doc.metadata["source"], "test.txt")
        self.assertEqual(lc_doc.metadata["chunk_index"], 2)
        self.assertEqual(lc_doc.metadata["distance"], 0.85)

    def test_2_rag_prompt_template(self):
        # TEST 2: ChatPromptTemplate correctly produces the RAG prompt.
        template = get_rag_prompt_template()
        prompt_val = template.invoke({"context": "doc info", "query": "what is it?"})
        prompt_str = prompt_val.to_string()
        
        self.assertIn("doc info", prompt_str)
        self.assertIn("what is it?", prompt_str)
        self.assertIn("answer the query using ONLY the provided context", prompt_str)

    def test_3_reflection_prompt_template(self):
        # TEST 3: Reflection prompt template preserves evidence-only instructions.
        template = get_reflection_prompt_template()
        prompt_val = template.invoke({"query": "q", "evidence": "ev"})
        prompt_str = prompt_val.to_string()
        
        self.assertIn("Evaluate ONLY the supplied evidence", prompt_str)
        self.assertIn("q", prompt_str)
        self.assertIn("ev", prompt_str)

    @patch("langchain_integration.get_vector_store")
    @patch("langchain_integration.embed_chunks")
    def test_4_5_retriever_adapter(self, mock_embed_chunks, mock_get_store):
        # TEST 4 & 5: Retriever adapter returns LangChain Documents & preserves metadata
        mock_embed_chunks.return_value = [[0.1, 0.2]]
        
        mock_store = MagicMock()
        mock_store.search.return_value = [
            CustomDocument(content="result 1", metadata={"source": "a.txt"})
        ]
        mock_get_store.return_value = mock_store
        
        retriever = LangChainRetrieverAdapter(k=1)
        # Using run_manager mock for invoke if needed, or invoke directly
        # langchain core retrievers expose invoke()
        results = retriever.invoke("search query")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "result 1")
        self.assertEqual(results[0].metadata["source"], "a.txt")
        self.assertEqual(type(results[0]).__name__, "Document")  # LangChain Document
        
    @patch("rag.store._get_backend")
    def test_6_rrf_fusion(self, mock_get_backend):
        # TEST 6: Existing RRF fusion still works
        from rag.store import FusionVectorStore
        import numpy as np
        
        mock_chroma = MagicMock()
        mock_pinecone = MagicMock()
        
        # Simple mock for get_backend
        def side_effect(name):
            if name == "chroma": return mock_chroma
            if name == "pinecone": return mock_pinecone
            return None
        mock_get_backend.side_effect = side_effect
        
        mock_chroma.search.return_value = [CustomDocument(content="A", metadata={"source": "A", "chunk_index": 0})]
        mock_pinecone.search.return_value = [CustomDocument(content="A", metadata={"source": "A", "chunk_index": 0})]
        
        fusion = FusionVectorStore()
        results = fusion.search(np.array([0.1]), k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "A")

    @patch("rag.store._get_backend")
    def test_7_retrieval_fallback(self, mock_get_backend):
        # TEST 7: Existing retrieval fallback still works
        from rag.store import FallbackVectorStore
        from rag.errors import RetryableRetrievalError
        import numpy as np
        
        mock_chroma = MagicMock()
        mock_pinecone = MagicMock()
        
        def side_effect(name):
            if name == "chroma": return mock_chroma
            if name == "pinecone": return mock_pinecone
            return None
        mock_get_backend.side_effect = side_effect
        
        mock_chroma.search.side_effect = RetryableRetrievalError("Chroma down")
        mock_pinecone.search.return_value = [CustomDocument(content="Fallback", metadata={"source": "B"})]
        
        fallback = FallbackVectorStore()
        fallback.primary_name = "chroma"
        fallback.fallback_name = "pinecone"
        fallback.fallback_enabled = True
        
        results = fallback.search(np.array([0.1]), k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Fallback")

    def test_8_agent_graph(self):
        # TEST 8: Existing agent graph construction still works
        from graph import graph
        self.assertIsNotNone(graph)
        self.assertTrue(hasattr(graph, "invoke"))

    def test_9_structured_output(self):
        # TEST 9: Structured output path intentionally not adopted for LangChain 
        # because it would require langchain-google-genai and langchain-openai, violating the dependency constraint.
        # We test that the original Pydantic validation still exists and works.
        from quality import ResearchQualityReport, ClaimAssessment
        report = ResearchQualityReport(
            assessments=[ClaimAssessment(claim="x", supported=True, reason="y", confidence=0.9, supporting_sources=[], conflicting_sources=[])],
            unsupported_claims=[],
            conflicts_detected=False,
            overall_grounding_score=1.0
        )
        self.assertEqual(report.overall_grounding_score, 1.0)
        self.assertTrue(report.assessments[0].supported)

    def test_10_no_api_keys(self):
        # TEST 10: No API keys appear in generated prompts.
        import os
        dummy_key = "sk-dummy-key-12345"
        os.environ["GEMINI_API_KEY"] = dummy_key
        
        template = get_rag_prompt_template()
        prompt_val = template.invoke({"context": "doc info", "query": "what is it?"})
        prompt_str = prompt_val.to_string()
        
        self.assertNotIn(dummy_key, prompt_str)

if __name__ == "__main__":
    unittest.main()
