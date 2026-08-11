import unittest
from unittest.mock import patch, MagicMock
from quality import (
    extract_claims, 
    assess_claims, 
    validate_citations, 
    detect_conflict,
    ClaimAssessment, 
    ResearchQualityReport
)
from research import EvidenceItem
from agent import execute_agent

class TestQuality(unittest.TestCase):

    def test_1_claim_extraction(self):
        answer = "RAG retrieves external information. It then provides that information to an LLM."
        claims = extract_claims(answer)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0], "RAG retrieves external information")
        self.assertEqual(claims[1], "It then provides that information to an LLM")

    def test_2_supported_claim(self):
        evidence = [EvidenceItem(content="RAG retrieves relevant information from an external knowledge base.", source="docs", source_type="local")]
        claims = ["RAG retrieves information from an external knowledge base."]
        report = assess_claims(claims, evidence)
        self.assertTrue(report.assessments[0].supported)

    def test_3_unsupported_claim(self):
        evidence = [EvidenceItem(content="RAG retrieves relevant information.", source="docs", source_type="local")]
        claims = ["RAG was invented in 2019."]
        report = assess_claims(claims, evidence)
        self.assertFalse(report.assessments[0].supported)

    def test_4_grounding_score(self):
        evidence = [EvidenceItem(content="RAG is fast. LLMs are cool. Python is great.", source="docs", source_type="local")]
        claims = ["RAG is fast.", "LLMs are cool.", "Python is great.", "Java is best."]
        report = assess_claims(claims, evidence)
        self.assertEqual(report.overall_grounding_score, 0.75)

    def test_5_citation_validation(self):
        evidence = [
            EvidenceItem(content="A", source="rag_overview.txt", source_type="local"),
            EvidenceItem(content="B", source="fine_tuning_overview.txt", source_type="local")
        ]
        answer = "This is a claim [LOCAL: rag_overview.txt] and another [LOCAL: unknown.txt]"
        invalid = validate_citations(answer, evidence)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0], "unknown.txt")

    def test_6_conflict_detection(self):
        evidence = [
            EvidenceItem(content="RAG requires fine-tuning.", source="web1", source_type="web"),
            EvidenceItem(content="RAG does not require fine-tuning.", source="web2", source_type="web")
        ]
        claims = ["RAG requires fine-tuning."]
        report = assess_claims(claims, evidence)
        self.assertTrue(len(report.conflicts_detected) > 0)

    def test_7_mixed_source_evidence(self):
        evidence = [
            EvidenceItem(content="Local data.", source="local1", source_type="local"),
            EvidenceItem(content="Web data.", source="web1", source_type="web")
        ]
        claims = ["Local data is web data."]
        report = assess_claims(claims, evidence)
        self.assertIsNotNone(report)

    def test_8_empty_evidence(self):
        report = assess_claims(["Some claim."], [])
        self.assertFalse(report.assessments[0].supported)
        self.assertEqual(report.overall_grounding_score, 0.0)

    @patch("providers.get_provider")
    def test_9_agent_integration(self, mock_get_provider):
        from providers import AgentResponse
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_response = AgentResponse(
            text="This is a basic answer.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "dummy"}
        )
        mock_provider.generate_agent_step.return_value = mock_response

        # Execute agent
        ans, state = execute_agent("Analyze this data.")
        
        # Verify research_quality is present
        self.assertIsNotNone(state.research_quality)
        mock_provider.generate.return_value = "Mocked grounding gate response"
        
    @patch("providers.get_provider")
    def test_10_refinement_guard(self, mock_get_provider):
        from providers import AgentResponse
        
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Return unsupported claim
        mock_response_1 = AgentResponse(
            text="RAG was invented in 2025.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "dummy1"}
        )
        
        # Then next iteration
        mock_response_2 = AgentResponse(
            text="RAG was invented recently.",
            function_calls=[],
            model_message={"role": "model", "raw_message": "dummy2"}
        )
        
        mock_provider.generate_agent_step.side_effect = [mock_response_1, mock_response_2]

        ans, state = execute_agent("When was RAG invented?")
        
        # Verify it went through refinement
        self.assertTrue(getattr(state, "refinement_attempted", False))
        
        # Wait, how many generate_content calls?
        self.assertEqual(mock_provider.generate_agent_step.call_count, 2)
        mock_provider.generate.return_value = "Mocked grounding gate response"

    @patch("providers.get_provider")
    def test_11_apply_grounding_gate(self, mock_get_provider):
        from quality import apply_grounding_gate
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Test 1: Supported claim survives (no rewriting triggered because score is 1.0)
        report1 = ResearchQualityReport(
            assessments=[ClaimAssessment("Supported claim", True, 0.9, ["web1"], [], "Supported")],
            overall_grounding_score=1.0,
            unsupported_claims=[],
            conflicts_detected=[]
        )
        ans1 = apply_grounding_gate("Supported claim", report1)
        self.assertEqual(ans1, "Supported claim")
        self.assertEqual(mock_provider.generate.call_count, 0)
        
        # Test 2: Unsupported claim triggers rewrite
        mock_provider.generate.return_value = "Rewritten answer without unsupported claims."
        report2 = ResearchQualityReport(
            assessments=[ClaimAssessment("Unsupported claim", False, 0.1, [], [], "Unsupported")],
            overall_grounding_score=0.0,
            unsupported_claims=["Unsupported claim"],
            conflicts_detected=[]
        )
        ans2 = apply_grounding_gate("Unsupported claim", report2)
        self.assertEqual(ans2, "Rewritten answer without unsupported claims.")
        self.assertEqual(mock_provider.generate.call_count, 1)

    def test_12_extract_claims_tables(self):
        answer = "Here is a table:\n| Company | Status |\n|---|---|\n| Toyota | 2027 |\n| BYD | Unknown |"
        claims = extract_claims(answer)
        self.assertIn("| Toyota | 2027 |", claims)
        self.assertIn("| BYD | Unknown |", claims)
        self.assertNotIn("|---|---|", claims)

    def test_13_chunking_logic(self):
        from quality import _chunk_text
        text = "P1\n\nP2\n\nP3"
        chunks = _chunk_text(text, max_length=5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "P1")
        self.assertEqual(chunks[1], "P2\n\nP3")
        
        # Test large paragraphs
        text_large = "A" * 2000 + "\n\n" + "B" * 2000
        chunks_large = _chunk_text(text_large, max_length=1500)
        self.assertEqual(len(chunks_large), 2)

    def test_14_claim_relevance(self):
        from quality import _is_claim_relevant
        chunk = "Toyota is targeting solid-state batteries in 2027."
        claim1 = "Toyota is targeting solid-state batteries in 2027."
        claim2 = "Samsung SDI plans to begin mass production in 2027."
        self.assertTrue(_is_claim_relevant(claim1, chunk))
        self.assertFalse(_is_claim_relevant(claim2, chunk))

    @patch("providers.get_provider")
    def test_15_grounding_gate_oversized(self, mock_get_provider):
        from quality import apply_grounding_gate
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Simulating a large answer that gets chunked
        chunk1 = "Chunk 1 is about Toyota's targets.\n\n" * 20 # 680 chars
        chunk2 = "Chunk 2 is about Samsung SDI's results.\n\n" * 20 # 800 chars
        answer = chunk1 + "\n\n" + chunk2
        
        # Mock LLM to return rewritten text for each chunk
        mock_provider.generate.side_effect = ["Rewritten Chunk 1", "Rewritten Chunk 2"]
        
        report = ResearchQualityReport(
            assessments=[
                ClaimAssessment("Toyota's targets.", False, 0.1, [], [], "Unsupported"),
                ClaimAssessment("Samsung SDI's results.", False, 0.1, [], [], "Unsupported")
            ],
            overall_grounding_score=0.0,
            unsupported_claims=["Toyota's targets.", "Samsung SDI's results."],
            conflicts_detected=[]
        )
        
        ans = apply_grounding_gate(answer, report)
        
        # LLM should be called twice, once for each chunk
        self.assertEqual(mock_provider.generate.call_count, 2)
        self.assertIn("Rewritten Chunk 1", ans)
        self.assertIn("Rewritten Chunk 2", ans)

    @patch("providers.get_provider")
    def test_16_grounding_gate_skip_clean_chunks(self, mock_get_provider):
        from quality import apply_grounding_gate
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Answer with two paragraphs
        chunk1 = "This paragraph has an unsupported claim about BYD." * 40
        chunk2 = "This paragraph is perfectly fine and supported." * 40
        answer = chunk1 + "\n\n" + chunk2
        
        mock_provider.generate.return_value = "Rewritten BYD paragraph."
        
        report = ResearchQualityReport(
            assessments=[],
            overall_grounding_score=0.5,
            unsupported_claims=["unsupported claim about BYD."],
            conflicts_detected=[]
        )
        
        ans = apply_grounding_gate(answer, report)
        
        # LLM should be called only ONCE for the chunk with issues
        self.assertEqual(mock_provider.generate.call_count, 1)
        self.assertIn("Rewritten BYD paragraph.", ans)
        # The clean chunk should be passed through untouched
        self.assertIn(chunk2, ans)

if __name__ == "__main__":
    unittest.main()
