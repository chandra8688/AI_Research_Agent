import unittest
from research import EvidenceItem
from quality import validate_citations

class TestValidateCitations(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            EvidenceItem(content="Some info", source="local_file.txt", source_type="local", metadata={}),
            EvidenceItem(content="Web info", source="Web Page (http://example.com)", source_type="web", metadata={})
        ]

    def test_valid_citation(self):
        ans = "Here is a fact [LOCAL: local_file.txt] and another [WEB: Web Page (http://example.com)]."
        invalid = validate_citations(ans, self.evidence)
        self.assertEqual(invalid, [])

    def test_invalid_citation(self):
        ans = "Here is a fact [LOCAL: wrong_file.txt]."
        invalid = validate_citations(ans, self.evidence)
        self.assertEqual(invalid, ["wrong_file.txt"])

    def test_no_citation_with_evidence(self):
        ans = "Here is a fact with no citation."
        invalid = validate_citations(ans, self.evidence)
        self.assertEqual(invalid, ["No source citations were provided."])

    def test_no_evidence(self):
        ans = "Here is a fact with no citation."
        invalid = validate_citations(ans, [])
        self.assertEqual(invalid, [])

if __name__ == '__main__':
    unittest.main()
