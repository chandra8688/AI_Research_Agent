import unittest
from unittest.mock import patch, MagicMock

from llm import call_llm
from providers.errors import RetryableProviderError, FatalProviderError

class TestLLMFallback(unittest.TestCase):

    def setUp(self):
        # Create common mock providers
        self.mock_gemini = MagicMock()
        self.mock_openrouter = MagicMock()
        
        # We need a side_effect function for get_provider to return our mocks
        def get_provider_side_effect(name):
            if name == "gemini":
                return self.mock_gemini
            elif name == "openrouter":
                return self.mock_openrouter
            else:
                raise ValueError(f"Unsupported LLM_PROVIDER: '{name}'")
                
        self.get_provider_patcher = patch('providers.get_provider', side_effect=get_provider_side_effect)
        self.mock_get_provider = self.get_provider_patcher.start()

    def tearDown(self):
        self.get_provider_patcher.stop()

    @patch('config.settings')
    def test_1_primary_succeeds(self, mock_settings):
        # TEST 1: Primary Gemini succeeds.
        mock_settings.llm_primary_provider = "gemini"
        mock_settings.llm_fallback_enabled = True
        mock_settings.llm_fallback_provider = "openrouter"
        
        self.mock_gemini.generate.return_value = "Gemini Answer"
        
        answer = call_llm("Hello")
        
        self.assertEqual(answer, "Gemini Answer")
        self.mock_gemini.generate.assert_called_once_with("Hello")
        self.mock_openrouter.generate.assert_not_called()

    @patch('config.settings')
    def test_2_retryable_429_fallback(self, mock_settings):
        # TEST 2: Gemini produces a retryable 429 failure. OpenRouter succeeds.
        mock_settings.llm_primary_provider = "gemini"
        mock_settings.llm_fallback_enabled = True
        mock_settings.llm_fallback_provider = "openrouter"
        
        self.mock_gemini.generate.side_effect = RetryableProviderError("Gemini API Error: 429 RESOURCE_EXHAUSTED")
        self.mock_openrouter.generate.return_value = "OpenRouter Answer"
        
        answer = call_llm("Hello")
        
        self.assertEqual(answer, "OpenRouter Answer")
        self.mock_gemini.generate.assert_called_once_with("Hello")
        self.mock_openrouter.generate.assert_called_once_with("Hello")

    @patch('config.settings')
    def test_3_retryable_503_fallback(self, mock_settings):
        # TEST 3: Gemini produces a retryable 503 failure. OpenRouter succeeds.
        mock_settings.llm_primary_provider = "gemini"
        mock_settings.llm_fallback_enabled = True
        mock_settings.llm_fallback_provider = "openrouter"
        
        self.mock_gemini.generate.side_effect = RetryableProviderError("Gemini API Error: 503 Service Unavailable")
        self.mock_openrouter.generate.return_value = "OpenRouter Answer"
        
        answer = call_llm("Hello")
        
        self.assertEqual(answer, "OpenRouter Answer")
        self.mock_gemini.generate.assert_called_once_with("Hello")
        self.mock_openrouter.generate.assert_called_once_with("Hello")

    @patch('config.settings')
    def test_4_non_retryable_failure(self, mock_settings):
        # TEST 4: Gemini produces a non-retryable authentication/configuration failure.
        mock_settings.llm_primary_provider = "gemini"
        mock_settings.llm_fallback_enabled = True
        mock_settings.llm_fallback_provider = "openrouter"
        
        self.mock_gemini.generate.side_effect = FatalProviderError("Gemini API Error: 400 Bad Request")
        
        with self.assertRaises(FatalProviderError):
            call_llm("Hello")
            
        self.mock_gemini.generate.assert_called_once_with("Hello")
        self.mock_openrouter.generate.assert_not_called()

    @patch('config.settings')
    def test_5_both_providers_fail(self, mock_settings):
        # TEST 5: Gemini fails and OpenRouter also fails.
        mock_settings.llm_primary_provider = "gemini"
        mock_settings.llm_fallback_enabled = True
        mock_settings.llm_fallback_provider = "openrouter"
        
        self.mock_gemini.generate.side_effect = RetryableProviderError("Gemini API Error: 429")
        self.mock_openrouter.generate.side_effect = FatalProviderError("OpenRouter API Error: 401 Unauthorized")
        
        with self.assertRaises(RuntimeError) as context:
            call_llm("Hello")
            
        err_msg = str(context.exception)
        self.assertIn("Both providers failed", err_msg)
        self.assertIn("Primary error: Gemini API Error: 429", err_msg)
        self.assertIn("Fallback error: OpenRouter API Error: 401 Unauthorized", err_msg)
        
        self.mock_gemini.generate.assert_called_once_with("Hello")
        self.mock_openrouter.generate.assert_called_once_with("Hello")

    @patch('config.settings')
    def test_6_fallback_disabled(self, mock_settings):
        # TEST 6: Fallback disabled.
        mock_settings.llm_primary_provider = "gemini"
        mock_settings.llm_fallback_enabled = False
        
        self.mock_gemini.generate.side_effect = RetryableProviderError("Gemini API Error: 429")
        
        with self.assertRaises(RetryableProviderError):
            call_llm("Hello")
            
        self.mock_gemini.generate.assert_called_once_with("Hello")
        self.mock_openrouter.generate.assert_not_called()

    @patch('config.settings')
    def test_7_unknown_provider(self, mock_settings):
        # TEST 7: Unknown provider configuration.
        mock_settings.llm_primary_provider = "unknown_provider"
        mock_settings.llm_fallback_enabled = True
        
        with self.assertRaises(ValueError) as context:
            call_llm("Hello")
            
        self.assertIn("Unsupported LLM_PROVIDER", str(context.exception))

    @patch('config.settings')
    def test_8_backward_compatibility(self, mock_settings):
        # TEST 8: Backward compatibility.
        # If llm_primary_provider is not set, it should fallback to llm_provider
        mock_settings.llm_primary_provider = None
        mock_settings.llm_provider = "gemini"
        mock_settings.llm_fallback_enabled = False
        
        self.mock_gemini.generate.return_value = "Gemini Compatible"
        
        answer = call_llm("Test")
        self.assertEqual(answer, "Gemini Compatible")
        self.mock_gemini.generate.assert_called_once_with("Test")


if __name__ == '__main__':
    unittest.main()
