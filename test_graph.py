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
        
        ans, state = execute_agent_graph("Explain this concept")
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
        
        ans, state = execute_agent_graph("Explain RAG")
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
        
        ans, state = execute_agent_graph("Find the latest news")
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
        
        ans, state = execute_agent_graph("Explain RAG")
        self.assertEqual(state.reflection_attempts, 2)
        self.assertEqual(ans, "Now it works.")

    @patch("graph.TOOL_REGISTRY")
    def test_8_reflection_limit(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        """When max_reflection_attempts is reached, force_synthesis must produce a final answer
        WITHOUT going back to agent_decide. provider.generate() is called, not generate_agent_step."""
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        # Iteration 1: agent makes a search tool call
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG"})],
            model_message={"role": "model"}
        )
        # Iteration 2: agent makes another search tool call
        call_resp2 = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG follow-up"})],
            model_message={"role": "model"}
        )

        # generate_agent_step used only for tool-calling iterations
        mock_provider.generate_agent_step.side_effect = [call_resp, call_resp2]
        # generate() used by force_synthesis for the final answer
        mock_provider.generate.return_value = "Forced synthesis answer. [LOCAL: local]"
        mock_registry.get.return_value = MagicMock(return_value="[Evidence 1]Source: local (Chunk 0)Distance: 0.1\nText: Local result")

        fail_eval = MagicMock()
        fail_eval.sufficient = False
        fail_eval.reason = "Still insufficient."
        # All reflection calls return insufficient — limit (=2) will be hit on the second call
        mock_eval.return_value = fail_eval

        # max_reflection_attempts=2 (default), so:
        # reflection attempt 1 → insufficient, attempts remain → agent_decide → another search
        # reflection attempt 2 → insufficient, limit reached → force_synthesis → quality_check
        ans, state = execute_agent_graph("Explain RAG", max_iterations=10)

        self.assertEqual(state.reflection_attempts, 2)
        # force_synthesis produces the answer via provider.generate(), not generate_agent_step
        mock_provider.generate.assert_called_once()
        self.assertIn("Forced synthesis answer.", ans)
        # Verify force_synthesis trace event was emitted
        trace_types = [t.event_type for t in state.trace]
        self.assertIn("force_synthesis", trace_types)
        self.assertIn("reflection_limit_reached", trace_types)

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

    # -------------------------------------------------------------------------
    # New routing tests for reflection attempt limit enforcement
    # -------------------------------------------------------------------------

    @patch("graph.TOOL_REGISTRY")
    def test_13_reflection_insufficient_attempts_remain(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        """TEST A: insufficient + attempts remain → route back to agent_decide (more research)."""
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        # Iteration 1: search, reflection insufficient (attempt 1 of 2 → attempts remain)
        # Iteration 2: agent decides to produce final answer after re-instruction
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG"})],
            model_message={"role": "model"}
        )
        final_resp = AgentResponse(
            text="Final answer after refinement. [LOCAL: local]",
            function_calls=[],
            model_message={"role": "model"}
        )
        mock_provider.generate_agent_step.side_effect = [call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(return_value="[Evidence 1]Source: local (Chunk 0)Distance: 0.1\nText: Local result")

        # Only 1 reflection → attempt 1 < max_reflection_attempts(2) → routes back to agent_decide
        mock_eval.return_value = MagicMock(sufficient=False, reason="Only one source, need more.")

        ans, state = execute_agent_graph("Explain RAG")

        # Verify: agent_decide was called again (generate_agent_step called twice)
        self.assertEqual(mock_provider.generate_agent_step.call_count, 2)
        # Verify: force_synthesis was NOT triggered
        self.assertNotIn("force_synthesis", [t.event_type for t in state.trace])
        self.assertEqual(state.reflection_attempts, 1)
        self.assertEqual(ans, "Final answer after refinement. [LOCAL: local]")

    @patch("graph.TOOL_REGISTRY")
    def test_14_reflection_limit_enforced(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        """TEST B: insufficient + limit reached → force_synthesis, NOT agent_decide."""
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG"})],
            model_message={"role": "model"}
        )
        call_resp2 = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_local_knowledge", args={"query": "RAG 2"})],
            model_message={"role": "model"}
        )
        mock_provider.generate_agent_step.side_effect = [call_resp, call_resp2]
        mock_provider.generate.return_value = "Force-synthesized answer. [LOCAL: local]"
        mock_registry.get.return_value = MagicMock(return_value="[Evidence 1]Source: local (Chunk 0)Distance: 0.1\nText: Some local result")

        # Always insufficient → after 2 attempts, limit hit → force_synthesis
        mock_eval.return_value = MagicMock(sufficient=False, reason="Always insufficient.")

        ans, state = execute_agent_graph("Explain RAG", max_iterations=10)

        # force_synthesis must be in trace
        self.assertIn("force_synthesis", [t.event_type for t in state.trace])
        self.assertIn("reflection_limit_reached", [t.event_type for t in state.trace])
        # generate_agent_step should NOT have been called for synthesis
        # (only 2 calls for the 2 search iterations)
        self.assertEqual(mock_provider.generate_agent_step.call_count, 2)
        # provider.generate() must have been called by force_synthesis
        mock_provider.generate.assert_called_once()
        self.assertEqual(state.reflection_attempts, 2)
        self.assertIn("Force-synthesized answer.", ans)

    @patch("graph.TOOL_REGISTRY")
    def test_15_reflection_sufficient_normal_path(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        """TEST C: sufficient=True before limit → normal agent_decide → quality_check path."""
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_web", args={"query": "batteries"})],
            model_message={"role": "model"}
        )
        final_resp = AgentResponse(
            text="Here is the answer from sufficient evidence. [WEB: Battery (http://example.com)]",
            function_calls=[],
            model_message={"role": "model"}
        )
        mock_provider.generate_agent_step.side_effect = [call_resp, final_resp]
        mock_registry.get.return_value = MagicMock(
            return_value="[Result 1]\nTitle: Battery\nURL: http://example.com\nSnippet: Some battery info"
        )
        # First (and only) reflection: sufficient
        mock_eval.return_value = MagicMock(sufficient=True, reason="Evidence covers the topic.")

        ans, state = execute_agent_graph("What are solid-state batteries?")

        # force_synthesis must NOT be in trace — normal path used
        self.assertNotIn("force_synthesis", [t.event_type for t in state.trace])
        self.assertEqual(state.reflection_attempts, 1)
        self.assertEqual(ans, "Here is the answer from sufficient evidence. [WEB: Battery (http://example.com)]")
        # provider.generate() must NOT have been called — only generate_agent_step
        mock_provider.generate.assert_not_called()

    @patch("graph.TOOL_REGISTRY")
    def test_16_force_synthesis_has_accumulated_evidence(self, mock_registry, mock_eval, mock_extract, mock_get_provider):
        """TEST D: force_synthesis receives all accumulated evidence from all prior searches."""
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        # Two search tool calls accumulate evidence
        call_resp = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_web", args={"query": "toyota battery"})],
            model_message={"role": "model"}
        )
        call_resp2 = AgentResponse(
            text=None,
            function_calls=[ToolCall(name="search_web", args={"query": "quantumscape battery"})],
            model_message={"role": "model"}
        )
        mock_provider.generate_agent_step.side_effect = [call_resp, call_resp2]

        search_results = [
            "[Result 1]\nTitle: Toyota SSB\nURL: http://toyota.com\nSnippet: Toyota solid-state data",
            "[Result 2]\nTitle: QuantumScape\nURL: http://qs.com\nSnippet: QuantumScape solid-state data",
        ]
        mock_registry.get.return_value = MagicMock(side_effect=search_results)

        # Both reflections insufficient → limit hit at attempt 2
        mock_eval.return_value = MagicMock(sufficient=False, reason="Missing data.")

        captured_prompts = []
        def capture_generate(prompt):
            captured_prompts.append(prompt)
            return "Synthesis using all evidence. [WEB: Toyota SSB (http://toyota.com)]"
        mock_provider.generate.side_effect = capture_generate

        ans, state = execute_agent_graph("Compare battery companies", max_iterations=10)

        # force_synthesis was triggered
        self.assertIn("force_synthesis", [t.event_type for t in state.trace])
        # The synthesis prompt must contain evidence from BOTH searches
        self.assertEqual(len(captured_prompts), 1)
        synthesis_prompt = captured_prompts[0]
        self.assertIn("Toyota SSB", synthesis_prompt)
        self.assertIn("QuantumScape", synthesis_prompt)
        # Evidence accumulation: both web items present in state
        self.assertEqual(len(state.multi_source_evidence), 2)

    @patch("planning.is_simple_query")
    def test_17_fast_path(self, mock_is_simple, mock_eval, mock_extract, mock_get_provider):
        """TEST E: fast path for simple queries bypasses tools."""
        mock_extract.return_value = []
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_is_simple.return_value = True
        mock_provider.generate.return_value = "Fast answer."
        
        ans, state = execute_agent_graph("What is 2+2?", max_iterations=10)
        
        # Must have fast_llm_path in trace
        self.assertIn("fast_llm_path", [t.event_type for t in state.trace])
        self.assertEqual(ans, "Fast answer.")
        mock_provider.generate.assert_called_once()
        mock_provider.generate_agent_step.assert_not_called()

    @patch("planning.is_simple_query")
    def test_18_provider_error(self, mock_is_simple, mock_eval, mock_extract, mock_get_provider):
        """TEST F: Provider error bubbles up correctly."""
        from providers.errors import RetryableProviderError
        mock_is_simple.return_value = False
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_provider.generate_agent_step.side_effect = RetryableProviderError("Rate limit")
        
        with self.assertRaises(RetryableProviderError):
            execute_agent_graph("What is RAG?")

if __name__ == "__main__":
    unittest.main()
