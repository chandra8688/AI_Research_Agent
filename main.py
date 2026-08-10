import sys
from dotenv import load_dotenv
from agent import run_agent


def main():
    load_dotenv()

    # ------------------------------------------------------------------
    # Test 1: Research prompt — triggers search_web
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: Research prompt (search_web tool)")
    print("=" * 60)
    research_prompt = (
        "What are the main advantages of RAG over fine-tuning for enterprise LLMs? "
        "Search for information and summarise the key points."
    )
    print(f"Prompt: '{research_prompt}'\n")
    try:
        answer = run_agent(research_prompt)
        print(f"\nAnswer:\n{answer.strip()}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # ------------------------------------------------------------------
    # Test 2: Calculator regression — verify calculate_product still works
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 2: Calculator regression (calculate_product tool)")
    print("=" * 60)
    calc_prompt = "What is 6 multiplied by 7?"
    print(f"Prompt: '{calc_prompt}'\n")
    try:
        answer = run_agent(calc_prompt)
        print(f"\nAnswer: {answer.strip()}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # ------------------------------------------------------------------
    # Test 3: search_web failure handling — empty query
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TEST 3: search_web failure handling (direct call, empty query)")
    print("=" * 60)
    from tools import search_web
    result = search_web(query="", max_results=3)
    print(f"Result: {result}\n")


if __name__ == "__main__":
    main()
