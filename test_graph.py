import unittest
import os
from unittest.mock import patch, MagicMock

from graph import execute_agent_graph, GraphState
from state import AgentState
from memory import ConversationMemory, AgentSession, create_session
from research import EvidenceItem

from config import settings
settings.gemini_api_key = "test_key"
os.environ["GEMINI_API_KEY"] = "test_key"

from providers import AgentResponse, ToolCall

@patch("providers.get_provider")
@patch("graph.extract_claims")
@patch("graph.evaluate_evidence")
class TestGraph(unittest.TestCase):
    
    def test_1_graph_construction(self, mock_eval, mock_extract, mock_get_provider):
        self.assertIsNotNone(execute_agent_graph)
        
    def test_2_simple_query(self, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_response = AgentResponse(
            text="This is a simple answer.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "dummy"}
        )
        mock_provider.generate_agent_step.return_value = mock_response
        
        ans, state = execute_agent_graph("What is 2+2?")
        self.assertEqual(ans, "This is a simple answer.")
        self.assertEqual(state.iteration, 1)

    @patch("graph.TOOL_REGISTRY")
    def test_3_calculator_query(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_eval.return_value = MagicMock(sufficient=True, reason="")
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="calculate_product", args={"a": 2, "b": 3})],
            model_message={"role": "model", "raw_message": "call"}
        )
        
        final_resp = AgentResponse(
            text="The product is 6.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "ans"}
        )
        
        mock_provider.generate_agent_step.side_effect = [call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(return_value=6)
        
        ans, state = execute_agent_graph("Calculate 2 * 3")
        self.assertEqual(ans, "The product is 6.")
        self.assertEqual(len(state.tool_calls), 1)

    @patch("graph.TOOL_REGISTRY")
    def test_4_local_rag(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_eval.return_value = MagicMock(sufficient=True, reason="")
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG"})],
            model_message={"role": "model", "raw_message": "call"}
        )
        
        final_resp = AgentResponse(
            text="RAG is great.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "ans"}
        )
        
        mock_provider.generate_agent_step.side_effect = [call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(return_value="[LOCAL] RAG is great.")
        
        ans, state = execute_agent_graph("What is RAG?")
        self.assertEqual(ans, "RAG is great.")
        self.assertEqual(state.tool_calls[0]["name"], "search_local_knowledge")

    @patch("graph.TOOL_REGISTRY")
    def test_5_web_query(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_eval.return_value = MagicMock(sufficient=True, reason="")
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_web", args={"query": "News"})],
            model_message={"role": "model", "raw_message": "call"}
        )
        
        final_resp = AgentResponse(
            text="Latest news.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "ans"}
        )
        
        mock_provider.generate_agent_step.side_effect = [call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(return_value="[WEB] Latest news.")
        
        ans, state = execute_agent_graph("What is the latest news?")
        self.assertEqual(ans, "Latest news.")
        self.assertEqual(state.tool_calls[0]["name"], "search_web")

    @patch("planning.create_research_plan")
    @patch("graph.TOOL_REGISTRY")
    def test_6_comparative_research(self, mock_registry, mock_plan, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_eval.return_value = MagicMock(sufficient=True, reason="")
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        plan_mock = MagicMock()
        plan_mock.requires_multi_source_research = True
        plan_mock.requires_local_knowledge = True
        plan_mock.requires_web = True
        mock_plan.return_value = plan_mock

        call1 = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG local"})],
            model_message={"role": "model"}
        )
        
        call2 = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_web", args={"query": "RAG web"})],
            model_message={"role": "model"}
        )
        
        final_resp = AgentResponse(
            text="Combined.",
            function_calls=[],
            model_message={"role": "model"}
        )
        
        mock_provider.generate_agent_step.side_effect = [call1, call2, final_resp]
        
        def tool_side_effect(**kwargs):
            if "query" in kwargs and "web" in kwargs["query"]:
                return "Title: RAG web\nURL: http\nSnippet: Web RAG"
            return "Local RAG chunk"
            
        mock_registry.get.return_value = tool_side_effect
        
        ans, state = execute_agent_graph("Compare local and web RAG")
        self.assertEqual(len(state.tool_calls), 2)
        self.assertEqual(state.tool_calls[0]["name"], "search_local_knowledge")
        self.assertEqual(state.tool_calls[1]["name"], "search_web")

    @patch("graph.TOOL_REGISTRY")
    def test_7_reflection(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG"})],
            model_message={"role": "model"}
        )
        
        final_resp = AgentResponse(
            text="Now it works.",
            function_calls=[],
            model_message={"role": "model"}
        )
        
        mock_provider.generate_agent_step.side_effect = [call_resp, call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(return_value="Local result")
        
        fail_eval = MagicMock()
        fail_eval.sufficient = False
        fail_eval.reason = "Need more."
        
        succ_eval = MagicMock()
        succ_eval.sufficient = True
        succ_eval.reason = "Good."
        
        mock_eval.side_effect = [fail_eval, succ_eval]
        
        ans, state = execute_agent_graph("What is RAG?")
        self.assertEqual(state.reflection_attempts, 2)
        self.assertEqual(ans, "Now it works.")

    @patch("graph.TOOL_REGISTRY")
    def test_8_reflection_limit(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG"})],
            model_message={"role": "model"}
        )
        
        final_resp = AgentResponse(
            text="I tried but failed.",
            function_calls=[],
            model_message={"role": "model"}
        )
        
        # Max reflections=2. So call_resp (iter 1), call_resp (iter 2), final_resp (iter 3)
        mock_provider.generate_agent_step.side_effect = [call_resp, call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(return_value="Local result")
        
        fail_eval = MagicMock()
        fail_eval.sufficient = False
        fail_eval.reason = "Need more."
        
        mock_eval.return_value = fail_eval
        
        ans, state = execute_agent_graph("What is RAG?", max_iterations=6)
        self.assertEqual(state.reflection_attempts, 2)
        self.assertEqual(ans, "I tried but failed.")

    def test_9_quality_check(self, mock_eval, mock_extract, mock_get_provider):
        mock_eval.return_value = MagicMock(sufficient=True, reason="")
        # First extraction returns unsupported claims, second returns empty (valid)
        mock_extract.side_effect = [["Unsupported"], []]
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        ans1 = AgentResponse(
            text="This claim is bad and unsupported.",
            function_calls=[],
            model_message={"role": "model"}
        )
        
        ans2 = AgentResponse(
            text="This claim is better.",
            function_calls=[],
            model_message={"role": "model"}
        )
        
        mock_provider.generate_agent_step.side_effect = [ans1, ans2]
        
        ans, state = execute_agent_graph("Tell me something unsupported.")
        self.assertTrue(getattr(state, "refinement_attempted", False))
        self.assertEqual(ans, "This claim is better.")

    def test_10_memory(self, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        resp = AgentResponse(
            text="Session active.",
            function_calls=[],
            model_message={"role": "model"}
        )
        mock_provider.generate_agent_step.return_value = resp
        
        session = create_session()
        
        ans1, state1 = execute_agent_graph("Hello", session=session)
        ans2, state2 = execute_agent_graph("World", session=session)
        
        self.assertEqual(len(session.memory.get_messages()), 4)

    def test_11_guardrails(self, mock_eval, mock_extract, mock_get_provider):
        with self.assertRaises(ValueError):
            execute_agent_graph("")
            
        with self.assertRaises(ValueError):
            execute_agent_graph("test", max_iterations=0)
            
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="fake_tool", args={})],
            model_message={"role": "model"}
        )
        
        mock_provider.generate_agent_step.return_value = call_resp
        
        with self.assertRaises(RuntimeError):
            execute_agent_graph("test")

    def test_12_trace(self, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        resp = AgentResponse(
            text="Trace test.",
            function_calls=[],
            model_message={"role": "model"}
        )
        mock_provider.generate_agent_step.return_value = resp
        
        ans, state = execute_agent_graph("test")
        
        types = [t.event_type for t in state.trace]
        self.assertIn("graph_start", types)
        self.assertIn("agent_start", types)
        self.assertIn("graph_end", types)

if __name__ == "__main__":
    unittest.main()
