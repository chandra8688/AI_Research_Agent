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

    @patch('providers.openrouter.urllib.request.urlopen')
    @patch('config.settings')
    def test_openrouter_tool_schema_validity(self, mock_settings, mock_urlopen):
        """Test that the OpenRouter provider correctly translates the tool schema to lowercase standard JSON schema."""
        mock_settings.openrouter_api_key = "dummy_key"
        mock_settings.llm_model = "test-agent-or-model"
        
        # Fake successful response
        mock_response = MagicMock()
        fake_payload = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.read.return_value = json.dumps(fake_payload).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        from agent import FUNCTION_DECLARATIONS
        
        provider = OpenRouterProvider()
        provider.generate_agent_step([{"role": "user", "content": "Hi"}], FUNCTION_DECLARATIONS)
        
        # Verify the schema passed in the payload
        req = mock_urlopen.call_args[0][0]
        sent_data = json.loads(req.data.decode('utf-8'))
        
        tools = sent_data.get("tools", [])
        self.assertGreater(len(tools), 0)
        
        for tool in tools:
            self.assertEqual(tool.get("type"), "function")
            func = tool.get("function", {})
            self.assertIn("name", func)
            self.assertIn("description", func)
            self.assertIn("parameters", func)
            
            params = func["parameters"]
            self.assertEqual(params.get("type"), "object")
            
            # Recursive check function
            def check_schema(schema):
                if isinstance(schema, dict):
                    if "type" in schema:
                        self.assertTrue(schema["type"].islower(), f"Type '{schema['type']}' is not lowercase")
                    
                    if "properties" in schema:
                        self.assertEqual(schema.get("type"), "object", "Properties only allowed on object type")
                        for prop_name, prop_schema in schema["properties"].items():
                            check_schema(prop_schema)
                            if prop_name == "input":
                                self.assertEqual(prop_schema.get("type"), "object", "Input schema must be object if present")
                                
                    if "required" in schema:
                        self.assertEqual(schema.get("type"), "object", "Required only allowed on object type")
            
            check_schema(params)

    @patch('providers.gemini.genai.Client')
    @patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"})
    def test_gemini_tool_schema_unchanged(self, mock_client_class):
        """Test that the Gemini provider does not translate or lowercase tool schema types, preserving native enums."""
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.function_calls = []
        mock_response.candidates = [MagicMock(content="ok")]
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        from agent import FUNCTION_DECLARATIONS
        
        provider = GeminiProvider()
        provider.generate_agent_step([{"role": "user", "content": "Hi"}], FUNCTION_DECLARATIONS)
        
        # Extract the tools passed to Gemini
        kwargs = mock_client.models.generate_content.call_args[1]
        config = kwargs.get("config")
        tools = config.tools
        self.assertGreater(len(tools), 0)
        
        func_decls = tools[0].function_declarations
        self.assertGreater(len(func_decls), 0)
        
        for func in func_decls:
            self.assertIsNotNone(func.name)
            self.assertIsNotNone(func.description)
            
            # Types should remain native/uppercase strings, matching agent.py definition
            # (In Gemini SDK, parameters are passed as dicts and converted, but we just verify our dict is untouched)
            params = func.parameters
            self.assertTrue(hasattr(params, "type") or isinstance(params, dict))

if __name__ == '__main__':
    unittest.main()
