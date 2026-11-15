"""Regression tests for bugs found during the v1.1 audit.

Each test pins a specific failure that previously crashed or misbehaved.
All tests run offline (providers and tools are mocked).
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from config import settings
settings.gemini_api_key = "test_key"

from graph import execute_agent_graph
from planning import ResearchPlan, create_research_plan, is_simple_query
from providers import AgentResponse, ToolCall
from providers.errors import RetryableProviderError
from research import EvidenceItem

LOCAL_RESULT = "[Evidence 1]\nSource: rag_overview.txt (Chunk 0)\nDistance: 0.1\nText: RAG retrieves documents."
WEB_RESULT = "[Result 1]\nTitle: RAG News\nURL: https://example.com/rag\nSnippet: RAG is evolving quickly."


def _multi_source_plan(query):
    return ResearchPlan(
        original_query=query, intent="comparative_research",
        requires_web=True, requires_local_knowledge=True,
        requires_calculation=False, requires_multi_source_research=True,
    )


def _tool_call(name, query, call_id=None):
    return ToolCall(name=name, args={"query": query}, id=call_id)


@patch("providers.get_provider")
@patch("graph.extract_claims", return_value=[])
@patch("graph.evaluate_evidence")
class TestGraphRegressions(unittest.TestCase):

    @patch("graph.create_research_plan", side_effect=_multi_source_plan)
    @patch("graph.TOOL_REGISTRY")
    def test_multi_source_partial_evidence_does_not_crash(self, mock_registry, _plan, mock_eval, _extract, mock_get_provider):
        """Previously collect_evidence referenced an undefined `res_mod` (NameError)
        whenever multi-source research had only collected one channel."""
        provider = MagicMock()
        mock_get_provider.return_value = provider
        mock_eval.return_value = MagicMock(sufficient=True, reason="ok")
        provider.generate_agent_step.side_effect = [
            AgentResponse(text=None, function_calls=[_tool_call("search_local_knowledge", "rag")], model_message={"role": "model"}),
            AgentResponse(text=None, function_calls=[_tool_call("search_web", "rag")], model_message={"role": "model"}),
            AgentResponse(text="Final [LOCAL: rag_overview.txt]", function_calls=[], model_message={"role": "model"}),
        ]
        tools = {"search_local_knowledge": MagicMock(return_value=LOCAL_RESULT),
                 "search_web": MagicMock(return_value=WEB_RESULT)}
        mock_registry.get.side_effect = tools.get

        answer, state = execute_agent_graph("Compare my local documents with the latest web news on RAG")

        self.assertIn("Final", answer)
        self.assertEqual({e.source_type for e in state.multi_source_evidence}, {"local", "web"})
        # Reflection only runs once both channels are present.
        self.assertEqual(mock_eval.call_count, 1)

    @patch("graph.TOOL_REGISTRY")
    def test_parallel_tool_calls_all_executed_and_answered(self, mock_registry, mock_eval, _extract, mock_get_provider):
        """Previously only function_calls[0] ran, but the model turn contained every call,
        so providers rejected the next request (call/response count mismatch)."""
        provider = MagicMock()
        mock_get_provider.return_value = provider
        mock_eval.return_value = MagicMock(sufficient=True, reason="ok")
        provider.generate_agent_step.side_effect = [
            AgentResponse(text=None, function_calls=[
                _tool_call("search_local_knowledge", "rag", "call_a"),
                _tool_call("search_web", "rag", "call_b"),
            ], model_message={"role": "model"}),
            AgentResponse(text="Done [WEB: RAG News (https://example.com/rag)]", function_calls=[], model_message={"role": "model"}),
        ]
        tools = {"search_local_knowledge": MagicMock(return_value=LOCAL_RESULT),
                 "search_web": MagicMock(return_value=WEB_RESULT)}
        mock_registry.get.side_effect = tools.get

        _, state = execute_agent_graph("Explain RAG")

        self.assertEqual([c["name"] for c in state.tool_calls], ["search_local_knowledge", "search_web"])
        responses = [m for m in state.contents if "function_responses" in m]
        self.assertEqual(len(responses), 1)
        parts = responses[0]["function_responses"]
        self.assertEqual([p["id"] for p in parts], ["call_a", "call_b"])

    @patch("planning.is_simple_query", return_value=True)
    def test_fast_path_provider_error_is_raised_cleanly(self, _simple, mock_eval, _extract, mock_get_provider):
        """Previously the fast path always continued to quality_check, which crashed
        with a KeyError instead of surfacing the provider error (e.g. a 429)."""
        provider = MagicMock()
        mock_get_provider.return_value = provider
        provider.generate.side_effect = RetryableProviderError("429 RESOURCE_EXHAUSTED")

        with self.assertRaises(RetryableProviderError):
            execute_agent_graph("What is the capital of France?")

    @patch("planning.is_simple_query", return_value=True)
    def test_fast_path_includes_conversation_history(self, _simple, mock_eval, _extract, mock_get_provider):
        from memory import create_session
        provider = MagicMock()
        mock_get_provider.return_value = provider
        provider.generate.side_effect = ["Ada Lovelace wrote the first program.", "She was born in 1815."]

        session = create_session()
        execute_agent_graph("Who is Ada Lovelace?", session=session)
        execute_agent_graph("When was she born?", session=session)

        second_prompt = provider.generate.call_args_list[1][0][0]
        self.assertIn("Ada Lovelace wrote the first program.", second_prompt)
        self.assertIn("When was she born?", second_prompt)

    @patch("graph.TOOL_REGISTRY")
    def test_iteration_limit_with_evidence_synthesises_instead_of_failing(self, mock_registry, mock_eval, _extract, mock_get_provider):
        """Hitting max_iterations after evidence was gathered used to return HTTP 500
        and throw the research away. It now synthesises from the collected evidence."""
        provider = MagicMock()
        mock_get_provider.return_value = provider
        mock_eval.return_value = MagicMock(sufficient=True, reason="ok")
        provider.generate_agent_step.return_value = AgentResponse(
            text=None, function_calls=[_tool_call("search_web", "rag")], model_message={"role": "model"})
        provider.generate.return_value = "Synthesised [WEB: RAG News (https://example.com/rag)]"
        mock_registry.get.return_value = MagicMock(return_value=WEB_RESULT)

        answer, state = execute_agent_graph("Explain RAG", max_iterations=2)

        self.assertIn("Synthesised", answer)
        self.assertIn("iteration_limit_reached", [t.event_type for t in state.trace])

    @patch("graph.TOOL_REGISTRY")
    def test_reflection_failure_does_not_fail_request(self, mock_registry, mock_eval, _extract, mock_get_provider):
        provider = MagicMock()
        mock_get_provider.return_value = provider
        mock_eval.side_effect = RuntimeError("judge returned invalid JSON")
        provider.generate_agent_step.side_effect = [
            AgentResponse(text=None, function_calls=[_tool_call("search_web", "rag")], model_message={"role": "model"}),
            AgentResponse(text="Answer [WEB: RAG News (https://example.com/rag)]", function_calls=[], model_message={"role": "model"}),
        ]
        mock_registry.get.return_value = MagicMock(return_value=WEB_RESULT)

        answer, state = execute_agent_graph("Explain RAG")

        self.assertIn("Answer", answer)
        self.assertIn("reflection_error", [t.event_type for t in state.trace])

    def test_long_research_loop_does_not_hit_langgraph_recursion_limit(self, mock_eval, _extract, mock_get_provider):
        """Older LangGraph releases default to a 25-step recursion limit, which is lower
        than what our iteration limits allow. The graph now sets an explicit limit so
        our own guardrails, not the library default, decide when to stop."""
        provider = MagicMock()
        mock_get_provider.return_value = provider
        mock_eval.return_value = MagicMock(sufficient=False, reason="more")
        provider.generate_agent_step.return_value = AgentResponse(
            text=None, function_calls=[ToolCall(name="calculate_product", args={"a": 2, "b": 3})],
            model_message={"role": "model"})

        with self.assertRaises(RuntimeError) as ctx:
            execute_agent_graph("Calculate 2 * 3", max_iterations=12)
        self.assertIn("did not produce a final answer", str(ctx.exception))

    @patch("graph.TOOL_REGISTRY")
    @patch("quality.apply_grounding_gate", side_effect=lambda text, *a, **k: text)
    def test_no_debug_file_written(self, _gate, mock_registry, mock_eval, mock_extract, mock_get_provider):
        mock_extract.return_value = ["An unsupported claim about something else entirely"]
        provider = MagicMock()
        mock_get_provider.return_value = provider
        mock_eval.return_value = MagicMock(sufficient=True, reason="ok")
        provider.generate_agent_step.side_effect = [
            AgentResponse(text=None, function_calls=[_tool_call("search_web", "rag")], model_message={"role": "model"}),
            AgentResponse(text="Draft [WEB: RAG News (https://example.com/rag)]", function_calls=[], model_message={"role": "model"}),
            AgentResponse(text="Revised [WEB: RAG News (https://example.com/rag)]", function_calls=[], model_message={"role": "model"}),
        ]
        mock_registry.get.return_value = MagicMock(return_value=WEB_RESULT)
        if os.path.exists("debug_answers.txt"):
            os.remove("debug_answers.txt")

        execute_agent_graph("Explain RAG")

        self.assertFalse(os.path.exists("debug_answers.txt"))


class TestCitationNormalisation(unittest.TestCase):

    def test_fullwidth_citation_brackets_are_normalised(self):
        """gpt-oss models answer with 【WEB: ...】, which the validator and UI ignored."""
        from graph import normalize_citation_brackets
        text = "Born in 1815【WEB: Ada Lovelace - Wikipedia (https://en.wikipedia.org/wiki/Ada_Lovelace)】."
        self.assertEqual(
            normalize_citation_brackets(text),
            "Born in 1815[WEB: Ada Lovelace - Wikipedia (https://en.wikipedia.org/wiki/Ada_Lovelace)].")
        self.assertEqual(normalize_citation_brackets("See 【LOCAL: rag_overview.txt】"), "See [LOCAL: rag_overview.txt]")
        self.assertEqual(normalize_citation_brackets("no citations"), "no citations")


class TestProviderMessageConversion(unittest.TestCase):

    def test_openai_messages_keep_history_and_map_parallel_ids(self):
        from providers._openai_compat import to_openai_messages
        messages = [
            {"role": "user", "content": "Who is Ada?"},
            {"role": "model", "content": "Ada Lovelace was a mathematician."},
            {"role": "user", "content": "Search more"},
            {"role": "model", "raw_message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "search_local_knowledge", "arguments": "{}"}},
            ]}},
            {"role": "user", "function_responses": [
                {"name": "search_web", "response": {"result": "a"}},
                {"name": "search_local_knowledge", "response": {"result": "b"}},
            ]},
        ]
        out = to_openai_messages(messages)
        self.assertEqual(out[1], {"role": "assistant", "content": "Ada Lovelace was a mathematician."})
        tool_msgs = [m for m in out if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["c1", "c2"])

    def test_gemini_keeps_plain_model_history(self):
        from providers.gemini import GeminiProvider
        provider = GeminiProvider()
        captured = {}

        class FakeModels:
            def generate_content(self, model, contents, config):
                captured["contents"] = contents
                resp = MagicMock()
                resp.function_calls = None
                resp.text = "ok"
                resp.candidates = []
                return resp

        with patch("providers.gemini.genai.Client") as client_cls:
            client_cls.return_value.models = FakeModels()
            provider.generate_agent_step([
                {"role": "user", "content": "Who is Ada?"},
                {"role": "model", "content": "Ada Lovelace was a mathematician."},
                {"role": "user", "content": "When was she born?"},
            ], [])

        roles = [c.role for c in captured["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])


class TestGroundingGateLayout(unittest.TestCase):

    @patch("providers.get_provider")
    def test_table_split_across_chunks_is_not_broken_apart(self, mock_get_provider):
        """Chunks used to be re-joined with blank lines, splitting a long markdown
        table into two (the second without a header, so it no longer rendered)."""
        from quality import apply_grounding_gate, ResearchQualityReport
        rows = "\n".join(f"| Company {i} | supported value number {i} |" for i in range(120))
        answer = "| Company | Value |\n|---|---|\n" + rows
        report = ResearchQualityReport(assessments=[], overall_grounding_score=0.0,
                                       unsupported_claims=["completely unrelated zebra claim"], conflicts_detected=[])
        result = apply_grounding_gate(answer, report)
        self.assertEqual(result, answer)
        mock_get_provider.return_value.generate.assert_not_called()


class TestPlannerKeywordMatching(unittest.TestCase):

    def test_hyphens_and_slashes_are_not_calculations(self):
        self.assertFalse(create_research_plan("Explain state-of-the-art RAG pipelines").requires_calculation)
        self.assertFalse(create_research_plan("Summarise input/output handling").requires_calculation)
        self.assertFalse(create_research_plan("What is the address of the API?").requires_calculation)

    def test_real_arithmetic_is_still_detected(self):
        self.assertTrue(create_research_plan("What is 42 * 17?").requires_calculation)
        self.assertTrue(create_research_plan("Multiply 12 by 9").requires_calculation)

    def test_substrings_do_not_trigger_keywords(self):
        self.assertFalse(create_research_plan("What is the currency of Japan?").requires_web)
        self.assertTrue(is_simple_query("What is a canvas element?"))

    def test_greetings_use_the_fast_path(self):
        """'Hii !' previously ran the full research + grounding pipeline."""
        for greeting in ["Hii !", "hi", "Hello there!", "thanks", "Thank you.", "good morning", "ok"]:
            self.assertTrue(is_simple_query(greeting), greeting)
        self.assertFalse(is_simple_query("Hi, what are the latest AI agent frameworks?"))

    def test_time_sensitive_questions_skip_the_fast_path(self):
        self.assertFalse(is_simple_query("Who is the current CEO of OpenAI?"))
        self.assertFalse(is_simple_query("Who is the president of France?"))
        self.assertFalse(is_simple_query("What happened in AI in 2025?"))


class TestConflictDetection(unittest.TestCase):

    def test_unrelated_numbers_are_not_conflicts(self):
        from quality import detect_conflict
        self.assertFalse(detect_conflict(
            "Toyota plans solid-state batteries by 2027",
            "QuantumScape raised 100 million for solid-state batteries"))

    def test_same_topic_different_number_is_conflict(self):
        from quality import detect_conflict
        self.assertTrue(detect_conflict(
            "Toyota plans solid-state batteries by 2027",
            "Toyota plans to launch solid-state batteries in 2028"))


class TestEmbedderCaching(unittest.TestCase):

    def test_model_loaded_once(self):
        import numpy as np
        from rag import embedder
        from rag.loader import Document
        embedder._get_model.cache_clear()
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda texts, convert_to_numpy: np.zeros((len(texts), 384))
        with patch.object(embedder, "SentenceTransformer", return_value=fake_model) as ctor:
            embedder.embed_chunks([Document(content="a")])
            embedder.embed_chunks([Document(content="b")])
        self.assertEqual(ctor.call_count, 1)
        embedder._get_model.cache_clear()


if __name__ == "__main__":
    unittest.main()
