import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock base environment to avoid real API keys leaking or failing
os.environ["GEMINI_API_KEY"] = "fake-secret-key"

from api_server import app
import api.routes
from config import settings

patch('api_server.initialize_knowledge_base').start()

class TestConfigAndAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        api.routes.sessions.clear()
        
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "VECTOR_DB": "chroma"})
    def test_default_configuration(self):
        from config import Settings
        settings = Settings(_env_file=None)
        self.assertEqual(settings.llm_provider, "gemini")
        self.assertEqual(settings.vector_db, "chroma")
        
    @patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"})
    def test_environment_override(self):
        from config import Settings
        custom_settings = Settings()
        self.assertEqual(custom_settings.llm_provider, "openrouter")
        
    def test_chroma_readiness_no_pinecone(self):
        # Even without pinecone API key, readiness should be ok if vector_db is chroma
        with patch('api.routes.settings.vector_db', 'chroma'):
            with patch('api.routes.settings.pinecone_api_key', None):
                response = self.client.get("/ready")
                self.assertEqual(response.status_code, 200)

    def test_missing_gemini_key_readiness(self):
        with patch('api.routes.settings.llm_provider', 'gemini'):
            with patch('api.routes.settings.gemini_api_key', None):
                response = self.client.get("/ready")
                self.assertEqual(response.status_code, 503)

    def test_missing_openrouter_key_readiness(self):
        with patch('api.routes.settings.llm_provider', 'openrouter'):
            with patch('api.routes.settings.openrouter_api_key', None):
                response = self.client.get("/ready")
                self.assertEqual(response.status_code, 503)

    def test_missing_pinecone_key_readiness(self):
        with patch('api.routes.settings.vector_db', 'pinecone'):
            with patch('api.routes.settings.pinecone_api_key', None):
                response = self.client.get("/ready")
                self.assertEqual(response.status_code, 503)
                
    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        
    def test_config_endpoint(self):
        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("environment", data)
        self.assertNotIn("gemini_api_key", data)
        self.assertNotIn("openrouter_api_key", data)
        
    def test_cors(self):
        # Validate wildcard is not used
        self.assertNotIn("*", settings.cors_origins)
        self.assertIn("http://localhost:3000", settings.cors_origins)
        
    def test_security(self):
        import subprocess
        result = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "")
        
    @patch('api.routes.execute_agent')
    def test_existing_api_regression(self, mock_execute_agent):
        mock_state = MagicMock()
        mock_state.iteration = 1
        mock_state.tool_calls = []
        mock_execute_agent.return_value = ("Test answer", mock_state)

        # POST /chat
        resp1 = self.client.post("/chat", json={"message": "Turn 1"})
        self.assertEqual(resp1.status_code, 200)
        session_id = resp1.json()["session_id"]
        
        # DELETE session
        resp2 = self.client.delete(f"/sessions/{session_id}")
        self.assertEqual(resp2.status_code, 200)

if __name__ == '__main__':
    unittest.main()
