import sys
from dotenv import load_dotenv
from agent import run_agent


def main():
    load_dotenv()

    # ------------------------------------------------------------------
    # Test 1: Zero tool calls — direct factual question
    # ------------------------------------------------------------------
    print("=" * 55)
    print("TEST 1: Zero-tool prompt")
    print("=" * 55)
    prompt_zero = "What is the capital of France?"
    print(f"Prompt: '{prompt_zero}'")
    try:
        answer = run_agent(prompt_zero)
        print(f"\nAnswer: {answer.strip()}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # ------------------------------------------------------------------
    # Test 2: Two-step tool chain
    # ------------------------------------------------------------------
    print("=" * 55)
    print("TEST 2: Two-step tool chain")
    print("=" * 55)
    prompt_chain = (
        "What is 6 multiplied by 7? "
        "Then multiply that result by 2. "
        "Give me the final number."
    )
    print(f"Prompt: '{prompt_chain}'")
    try:
        answer = run_agent(prompt_chain)
        print(f"\nAnswer: {answer.strip()}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # ------------------------------------------------------------------
    # Test 3: max_iterations guard (max=1 forces early termination)
    # ------------------------------------------------------------------
    print("=" * 55)
    print("TEST 3: max_iterations guard (max_iterations=1, chain of 2)")
    print("=" * 55)
    print(f"Prompt: '{prompt_chain}'")
    try:
        answer = run_agent(prompt_chain, max_iterations=1)
        print(f"Unexpected success: {answer}\n")
    except RuntimeError as e:
        print(f"[EXPECTED ERROR CAUGHT] {e}\n")


if __name__ == "__main__":
    main()
