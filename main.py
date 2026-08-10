import os
import sys
from dotenv import load_dotenv
from llm import call_llm

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    print("Testing Basic LLM Call...")
    prompt = "What is the capital of France? Reply in one short sentence."
    print(f"Prompt: '{prompt}'")
    
    try:
        response = call_llm(prompt)
        print("\n--- Response ---")
        print(response)
        print("----------------")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
