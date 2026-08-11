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

if __name__ == "__main__":
    unittest.main()
