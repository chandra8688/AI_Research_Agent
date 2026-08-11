import unittest
from unittest.mock import patch, MagicMock
from planning import create_research_plan
from agent import execute_agent
from state import AgentState

class TestPlanning(unittest.TestCase):

    def test_1_calculation(self):
        plan = create_research_plan("What is 42 * 17?")
        self.assertEqual(plan.intent, "calculation")
        self.assertTrue(plan.requires_calculation)
        self.assertFalse(plan.requires_web)
        self.assertFalse(plan.requires_local_knowledge)

    def test_2_local_rag(self):
        plan = create_research_plan("According to my local documentation, explain RAG.")
        self.assertEqual(plan.intent, "local_research")
        self.assertTrue(plan.requires_local_knowledge)
        self.assertFalse(plan.requires_web)
        self.assertFalse(plan.requires_calculation)

    def test_3_web_query(self):
        plan = create_research_plan("What are the latest developments in AI agents?")
        self.assertEqual(plan.intent, "web_research")
        self.assertTrue(plan.requires_web)
        self.assertFalse(plan.requires_local_knowledge)
        self.assertFalse(plan.requires_calculation)

    def test_4_mixed_research(self):
        plan = create_research_plan("Compare the latest RAG techniques with the approaches described in my local documents.")
        self.assertEqual(plan.intent, "comparative_research")
        self.assertTrue(plan.requires_web)
        self.assertTrue(plan.requires_local_knowledge)
        self.assertTrue(plan.requires_multi_source_research)
        self.assertFalse(plan.requires_calculation)

    def test_5_general_question(self):
        plan = create_research_plan("What is machine learning?")
        self.assertEqual(plan.intent, "general_knowledge")
        self.assertFalse(plan.requires_web)
        self.assertFalse(plan.requires_local_knowledge)
        self.assertFalse(plan.requires_calculation)
        self.assertFalse(plan.requires_multi_source_research)

    def test_6_empty_query(self):
        with self.assertRaises(ValueError):
            create_research_plan("   ")


    @patch("providers.get_provider")
    def test_7_agent_regression(self, mock_get_provider):
        from providers import AgentResponse
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_response = AgentResponse(
            text="RAG is...",
            function_calls=[],
            model_message={"role": "model", "raw_message": "dummy"}
        )
        mock_provider.generate_agent_step.return_value = mock_response

        # Execute agent
        ans, state = execute_agent("According to my files what is RAG?")
        self.assertEqual(ans, "RAG is...")
        
        # Check plan is attached to state
        self.assertIsNotNone(state.research_plan)
        self.assertEqual(state.research_plan.intent, "local_research")
        
        # Check plan was added to trace
        has_plan_trace = any(t.event_type == "research_plan" for t in state.trace)
        self.assertTrue(has_plan_trace)

if __name__ == "__main__":
    unittest.main()
