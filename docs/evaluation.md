# Project Evaluation & Regression History

This document outlines the strict validation standards and test boundaries applied to the AI Research Agent. Tests are firmly segregated between offline behavior mocking and live network interactions.

## 1. Offline Regression Suite (MOCK TESTS)

The project maintains an exhaustive suite of offline `unittest` frameworks. These tests execute strictly via patched internal dependencies, guaranteeing structural validity without consuming external quotas or requiring live API keys.

- **AI-240 Offline Verification:** `79 / 79 tests passed.` This validated the LangGraph orchestrator migration, ensuring planning loops, provider fallback logic, and tool extraction ran precisely as the prior manual loops did.
- **AI-250 Offline Verification:** `88 / 88 tests passed.` Following the introduction of LangChain Core abstractions (Documents, Prompts, Retriever adapters), the suite was expanded and successfully passed all existing architectural tests, verifying zero regression issues against the custom RAG logic.
- **Zero Live Calls (AI-250):** The AI-250 architecture integration was performed using strictly offline API mocks. Zero network calls were made to Gemini, OpenRouter, or Pinecone to validate the LangChain interfaces.

## 2. Live API Smoke Tests (LIVE TESTS)

During the AI-240 stabilization phase, live end-to-end (E2E) testing was executed against the primary provider (Gemini) to evaluate production behavior over standard HTTP endpoints.

### Real Gemini Test Observations
- **Test 1 (Local RAG):** Successfully routed an un-scoped query to the `search_local_knowledge` tool. The agent reflected on the raw output, structured the claims, and successfully returned a fully grounded response leveraging `[LOCAL]` markdown citations. Execution time: 30.79s.
- **Test 2 (Calculator):** Correctly bypassed RAG processing for a raw mathematical intent. Routed explicitly to `calculate_product` and returned exact arithmetic output (`714`). Execution time: 5.53s.

### Limitations & Quota Constraints
- **Tests 3 & 4 (Quota Hit):** The final evaluation tests (Multi-source research and Conversation Sessions) forcefully halted due to a `429 RESOURCE_EXHAUSTED` error from the Google Gemini API. 
- **Graceful Failure:** The backend successfully bubbled the rate limit into an HTTP 500 error instead of causing an infinite orchestration loop. The test was intentionally suspended at this point to preserve external quota boundaries and honor system safety instructions.
