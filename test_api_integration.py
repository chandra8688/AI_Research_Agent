from fastapi.testclient import TestClient
from api_server import app
from unittest.mock import patch, MagicMock
from state import AgentState, TraceEvent
import unittest

client = TestClient(app)

class TestApiIntegration(unittest.TestCase):
    
    def test_1_page_loads(self):
        # TEST 1 - Page loads
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.content)
        self.assertIn(b"AI RESEARCH AGENT", response.content)
        
    def test_2_health(self):
        # TEST 2 - Health
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        
    @patch('api.routes.execute_agent')
    def test_3_chat_sources_and_trace(self, mock_execute):
        # TEST 3 - Chat with sources and trace
        # Mock execute_agent to return an answer and a state with sources
        mock_state = AgentState(query="What is RAG?")
        mock_state.iteration = 1
        mock_state.tool_calls = [{"name": "search_local_knowledge"}]
        mock_state.retrieved_evidence = ["[Evidence 1]\nSource: docs.txt (Chunk 0)\nDistance: 0.1\nText: Data"]
        mock_state.trace = [
            TraceEvent(timestamp=0, event_type="agent_start", iteration=1, details={"query": "What is RAG?"}),
            TraceEvent(timestamp=1, event_type="tool_call", iteration=1, details={"tool_name": "search_local_knowledge"})
        ]
        
        mock_execute.return_value = ("RAG is Retrieval-Augmented Generation.", mock_state)
        
        response = client.post("/chat", json={"message": "What is RAG?"})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["answer"], "RAG is Retrieval-Augmented Generation.")
        self.assertIsNotNone(data["session_id"])
        
        # Verify sources parsed correctly
        self.assertIsNotNone(data["sources"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["source"], "docs.txt")
        self.assertEqual(data["sources"][0]["chunk_index"], "0")
        
        # Verify trace parsed correctly
        self.assertIsNotNone(data["trace"])
        self.assertEqual(len(data["trace"]), 2)
        self.assertEqual(data["trace"][0]["event_type"], "agent_start")
        self.assertEqual(data["trace"][1]["event_type"], "tool_call")

    @patch('api.routes.execute_agent')
    def test_4_error(self, mock_execute):
        # TEST 4 - Error
        mock_execute.side_effect = RuntimeError("Something broke")
        
        response = client.post("/chat", json={"message": "Break"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Something broke", response.json()["detail"])

    def test_5_empty_input(self):
        # TEST 5 - Empty input
        response = client.post("/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Message cannot be empty", response.json()["detail"])

    @patch('api.routes.execute_agent')
    def test_6_multi_turn(self, mock_execute):
        # TEST 6 - Multi-turn session preservation
        mock_state = AgentState(query="test")
        mock_execute.return_value = ("Answer 1", mock_state)
        
        res1 = client.post("/chat", json={"message": "Hello"})
        self.assertEqual(res1.status_code, 200)
        session_id = res1.json()["session_id"]
        
        mock_execute.return_value = ("Answer 2", mock_state)
        res2 = client.post("/chat", json={"message": "Follow up", "session_id": session_id})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["session_id"], session_id)
        
    @patch('api.routes.execute_agent')
    def test_7_security(self, mock_execute):
        # TEST 7 - Security (backend returns script)
        # The frontend uses JS textContent to prevent XSS.
        # We just verify the API passes the string correctly without mangling it,
        # so the frontend can escape it.
        mock_state = AgentState(query="test")
        malicious_answer = '<script>alert("x")</script>'
        mock_execute.return_value = (malicious_answer, mock_state)
        
        response = client.post("/chat", json={"message": "Inject"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], malicious_answer)

if __name__ == '__main__':
    unittest.main()
