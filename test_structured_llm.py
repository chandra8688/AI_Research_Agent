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
        
    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_structured_openrouter_schema_rejected(self, mock_settings, mock_urlopen):
        """Test OpenRouter rejects a JSON Schema definition returned instead of an instance."""
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-model"
        
        mock_response = MagicMock()
        # This matches the Nemotron failure pattern where it returns the schema definition
        fake_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "required": ["name", "age"]
                        })
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
            
        self.assertIn("JSON Schema definition instead of a populated instance", str(context.exception))

    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_structured_openrouter_prompt_instructions(self, mock_settings, mock_urlopen):
        """Test the prompt explicitly distinguishes schema definition from schema instance."""
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-or-model"
        
        mock_response = MagicMock()
        fake_payload = {
            "choices": [{"message": {"content": '{"name": "Test", "age": 1}'}}]
        }
        mock_response.read.return_value = json.dumps(fake_payload).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = OpenRouterProvider()
        provider.generate_structured("My Test Prompt", DummySchema)
        
        req = mock_urlopen.call_args[0][0]
        sent_data = json.loads(req.data.decode('utf-8'))
        prompt_sent = sent_data["messages"][0]["content"]
        
        self.assertIn("=== STRUCTURED OUTPUT INSTRUCTIONS ===", prompt_sent)
        self.assertIn("Do NOT return the JSON Schema definition", prompt_sent)
        self.assertIn("Return ONLY the filled-in JSON instance", prompt_sent)
        self.assertIn("Respond with the filled JSON instance now:", prompt_sent)
        self.assertIn('"name": <string value (required)>', prompt_sent)
        

    @patch('providers.groq.groq.Groq')
    @patch('config.settings')
    def test_structured_groq_path(self, mock_settings, mock_groq_class):
        mock_settings.groq_api_key = "dummy_key"
        mock_settings.groq_model = "test-groq-model"
        
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "GroqTest", "age": 3}'
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        from providers.groq import GroqProvider
        provider = GroqProvider()
        result = provider.generate_structured("Prompt", DummySchema)
        
        self.assertEqual(result.name, "GroqTest")
        self.assertEqual(result.age, 3)

if __name__ == '__main__':
    unittest.main()
