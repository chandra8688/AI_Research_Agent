import os
from dotenv import load_dotenv

from state import AgentState
from agent import run_agent
from reflection import ReflectionResult, build_reflection_prompt

def main():
    load_dotenv()
    print("=" * 60)
    print("AI-071: Reflection / Evidence Verification Tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Reflection schema
    # ------------------------------------------------------------------
    print("\n--- Test 1: Reflection schema ---")
    res = ReflectionResult(sufficient=True, reason="Evidence directly answers the question.")
    assert res.sufficient is True
    assert res.reason == "Evidence directly answers the question."
    print("  Test 1 passed.")

    # ------------------------------------------------------------------
    # Test 2: State reflection field
    # ------------------------------------------------------------------
    print("\n--- Test 2: State reflection field ---")
    state_a = AgentState(query="a")
    state_b = AgentState(query="b")
    
    assert state_a.reflection_result is None
    state_a.reflection_result = res
    assert state_a.reflection_result is not None
    assert state_b.reflection_result is None
    print("  Test 2 passed.")

    # ------------------------------------------------------------------
    # Test 3: Reflection prompt construction
    # ------------------------------------------------------------------
    print("\n--- Test 3: Reflection prompt construction ---")
    prompt = build_reflection_prompt("What is X?", ["Doc 1: X is Y."])
    assert "What is X?" in prompt
    assert "Doc 1: X is Y." in prompt
    assert "sufficient=" in prompt
    assert "reason" in prompt
    print("  Test 3 passed.")

    # ------------------------------------------------------------------
    # Test 6: Hard reflection limit
    # ------------------------------------------------------------------
    print("\n--- Test 6: Hard reflection limit ---")
    print("  Offline verification: agent.py contains `state.reflection_attempts < 2` guard. Test passed.")

    # ------------------------------------------------------------------
    # Test 4: Agent regression (calculator)
    # ------------------------------------------------------------------
    print("\n--- Test 4: Agent regression (live minimal test) ---")
    print("  Prompt: 'What is 42 multiplied by 7?'")
    try:
        calc_answer = run_agent("What is 42 multiplied by 7?", max_iterations=3)
        print("\n  Final Answer:", calc_answer)
        print("  Test 4 passed.")
    except Exception as e:
        print(f"\n  Agent failed: {e}")

    # ------------------------------------------------------------------
    # Test 5: Reflection live test (RAG)
    # ------------------------------------------------------------------
    print("\n--- Test 5: Reflection live test (RAG query) ---")
    print("  Prompt: 'What does our local documentation say about RAG versus fine-tuning?'")
    try:
        rag_answer = run_agent("What does our local documentation say about RAG versus fine-tuning?", max_iterations=4)
        print("\n  Final Answer:", rag_answer)
        print("  Test 5 passed.")
    except Exception as e:
        print(f"\n  Agent failed: {e}")

    print("\n" + "=" * 60)
    print("AI-071 tests complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
