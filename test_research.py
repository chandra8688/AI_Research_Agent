import unittest
from unittest.mock import patch, MagicMock
from research import EvidenceItem, combine_evidence, parse_local_evidence, parse_web_evidence, format_combined_evidence
from planning import create_research_plan
from state import AgentState
from agent import execute_agent
from google.genai import types

class TestResearchSynthesis(unittest.TestCase):
    
    def test_1_general_query(self):
        plan = create_research_plan("What is the capital of France?")
        self.assertFalse(plan.requires_web)
        self.assertFalse(plan.requires_local_knowledge)
        
    def test_2_local_only_query(self):
        plan = create_research_plan("According to my local files, what is the policy?")
        self.assertTrue(plan.requires_local_knowledge)
        self.assertFalse(plan.requires_web)
        
    def test_3_web_only_query(self):
        plan = create_research_plan("What are the latest developments in space?")
        self.assertTrue(plan.requires_web)
        self.assertFalse(plan.requires_local_knowledge)
        
    def test_4_comparative_query(self):
        plan = create_research_plan("Compare the latest news to my local documents.")
        self.assertTrue(plan.requires_web)
        self.assertTrue(plan.requires_local_knowledge)
        self.assertTrue(plan.requires_multi_source_research)
        
    def test_5_evidence_combination(self):
        local_result = "[Evidence 1]\nSource: docs.txt (Chunk 0)\nDistance: 0.1\nText: Local data"
        web_result = "[Result 1]\nTitle: Web News\nURL: http://news.com\nSnippet: Web data"
        
        items = combine_evidence(local_result, web_result)
        self.assertEqual(len(items), 2)
        
        self.assertEqual(items[0].source_type, "local")
        self.assertEqual(items[0].source, "docs.txt")
        self.assertEqual(items[0].content, "Local data")
        
        self.assertEqual(items[1].source_type, "web")
        self.assertEqual(items[1].source, "Web News (http://news.com)")
        self.assertEqual(items[1].content, "Web data")
        
    def test_6_source_attribution_preserved(self):
        items = [
            EvidenceItem(content="A", source="file.txt", source_type="local"),
            EvidenceItem(content="B", source="Site (url)", source_type="web")
        ]
        text = format_combined_evidence(items)
        self.assertIn("[LOCAL: file.txt]", text)
        self.assertIn("[WEB: Site (url)]", text)
        
    def test_7_conflicting_evidence_represented(self):
        # We verify that both pieces of evidence remain in the combined text distinctly.
        items = [
            EvidenceItem(content="Apples are red.", source="file.txt", source_type="local"),
            EvidenceItem(content="Apples are blue.", source="Site (url)", source_type="web")
        ]
        text = format_combined_evidence(items)
        self.assertIn("Apples are red.", text)
        self.assertIn("Apples are blue.", text)
        self.assertIn("[LOCAL: file.txt]", text)
        self.assertIn("[WEB: Site (url)]", text)

    @patch("providers.get_provider")
    def test_8_existing_guardrails_still_work(self, mock_get_provider):
        from providers import AgentResponse, ToolCall
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Test max iterations limit
        mock_response = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="calculate_product", args={"a": 2, "b": 3})],
            model_message={"role": "model", "raw_message": "dummy"}
        )
        mock_provider.generate_agent_step.return_value = mock_response

        # This will loop until max_iterations
        with self.assertRaises(RuntimeError) as context:
            execute_agent("What is 2 * 3?", max_iterations=2)
            
        self.assertIn("did not produce a final answer", str(context.exception))

if __name__ == "__main__":
    unittest.main()
