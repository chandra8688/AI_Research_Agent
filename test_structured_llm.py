import unittest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
import os
import json

from llm import call_llm_structured
from providers.gemini import GeminiProvider
from providers.openrouter import OpenRouterProvider
from providers.errors import FatalProviderError
from reflection import evaluate_evidence, ReflectionResult

class DummySchema(BaseModel):
    name: str
    age: int

class TestStructuredLLM(unittest.TestCase):

    @patch('providers.get_provider')
    @patch('config.settings')
    def test_provider_selection(self, mock_settings, mock_get_provider):
        """Test that structured LLM routing respects the LLM_PROVIDER configuration."""
        # Setup mock for primary
        mock_settings.llm_primary_provider = None
        mock_settings.llm_provider = "openrouter"
        
        mock_provider = MagicMock()
        mock_provider.generate_structured.return_value = DummySchema(name="Test", age=30)
        mock_get_provider.return_value = mock_provider
        
        result = call_llm_structured("Prompt", DummySchema)
        
        # Verify it fetched openrouter
        mock_get_provider.assert_called_once_with("openrouter")
        mock_provider.generate_structured.assert_called_once_with("Prompt", DummySchema)
        self.assertEqual(result.name, "Test")

    @patch('providers.gemini.genai.Client')
    @patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"})
    def test_structured_gemini_path(self, mock_client_class):
        """Test the structured generation works for Gemini natively."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.parsed = DummySchema(name="GeminiTest", age=1)
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider()
        result = provider.generate_structured("Prompt", DummySchema)
        
        self.assertEqual(result.name, "GeminiTest")
        self.assertEqual(result.age, 1)

    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_structured_openrouter_path(self, mock_settings, mock_urlopen):
        """Test OpenRouter successfully parses standard JSON."""
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-or-model"
        
        # Fake successful JSON response
        mock_response = MagicMock()
        fake_payload = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"name": "OpenRouterTest", "age": 2}\n```'
                    }
                }
            ]
        }
        mock_response.read.return_value = json.dumps(fake_payload).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = OpenRouterProvider()
        result = provider.generate_structured("Prompt", DummySchema)
        
        self.assertEqual(result.name, "OpenRouterTest")
        self.assertEqual(result.age, 2)
        
        # Verify model was passed
        req = mock_urlopen.call_args[0][0]
        sent_data = json.loads(req.data.decode('utf-8'))
        self.assertEqual(sent_data["model"], "test-or-model")

    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_structured_openrouter_malformed(self, mock_settings, mock_urlopen):
        """Test OpenRouter raises FatalProviderError on malformed JSON."""
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-model"
        
        mock_response = MagicMock()
        fake_payload = {
            "choices": [
                {
                    "message": {
                        "content": "This is not json."
                    }
                }
            ]
        }
        mock_response.read.return_value = json.dumps(fake_payload).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = OpenRouterProvider()
        with self.assertRaises(FatalProviderError) as context:
            provider.generate_structured("Prompt", DummySchema)
            
        self.assertIn("OpenRouter returned invalid JSON", str(context.exception))

    @patch('reflection.call_llm_structured')
    def test_reflection_provider_abstraction(self, mock_call_llm_structured):
        """Test reflection uses the provider abstraction properly."""
        mock_call_llm_structured.return_value = ReflectionResult(sufficient=True, reason="Looks good")
        
        result = evaluate_evidence("Who?", ["Evidence string"])
        
        self.assertTrue(result.sufficient)
        self.assertEqual(result.reason, "Looks good")
        mock_call_llm_structured.assert_called_once()
        
if __name__ == '__main__':
    unittest.main()
