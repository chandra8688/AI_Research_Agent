import unittest
from unittest.mock import patch, MagicMock
import os
import json

from providers.gemini import GeminiProvider
from providers.openrouter import OpenRouterProvider
from providers.errors import FatalProviderError
from providers import ToolCall

class TestAgentProviders(unittest.TestCase):

    @patch('providers.gemini.genai.Client')
    @patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"})
    def test_gemini_tool_calling(self, mock_client_class):
        mock_response = MagicMock()
        mock_response.text = None
        
        fc = MagicMock()
        fc.name = "search_web"
        fc.args = {"query": "AI"}
        mock_response.function_calls = [fc]
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        provider = GeminiProvider()
        result = provider.generate_agent_step([{"role": "user", "content": "Search AI"}], [])
        
        self.assertEqual(len(result.function_calls), 1)
        self.assertEqual(result.function_calls[0].name, "search_web")
        self.assertEqual(result.function_calls[0].args, {"query": "AI"})
        self.assertIsNone(result.text)

    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_openrouter_tool_calling(self, mock_settings, mock_urlopen):
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-agent-or-model"
        
        mock_response = MagicMock()
        fake_payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search_web",
                                    "arguments": '{"query": "AI"}'
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_response.read.return_value = json.dumps(fake_payload).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = OpenRouterProvider()
        result = provider.generate_agent_step([{"role": "user", "content": "Search AI"}], [])
        
        self.assertEqual(len(result.function_calls), 1)
        self.assertEqual(result.function_calls[0].name, "search_web")
        self.assertEqual(result.function_calls[0].args, {"query": "AI"})
        self.assertIsNone(result.text)
        
        # Verify model was passed
        req = mock_urlopen.call_args[0][0]
        sent_data = json.loads(req.data.decode('utf-8'))
        self.assertEqual(sent_data["model"], "test-agent-or-model")

    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_openrouter_malformed_response(self, mock_settings, mock_urlopen):
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-model"
        
        mock_response = MagicMock()
        # Invalid JSON returned by HTTP
        mock_response.read.return_value = b'Not JSON'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = OpenRouterProvider()
        with self.assertRaises(FatalProviderError):
            provider.generate_agent_step([{"role": "user", "content": "Search AI"}], [])

if __name__ == '__main__':
    unittest.main()
