import sys
import argparse
from dotenv import load_dotenv
from google.genai import errors
import time

from rag.pipeline import initialize_knowledge_base
from agent import execute_agent

def main():
    parser = argparse.ArgumentParser(description="AI Research Agent CLI")
    parser.add_argument("query", type=str, nargs="?", help="The research query to execute")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive session mode")
    args = parser.parse_args()

    if not args.interactive and not args.query:
        print("Usage: python main.py \"<your query>\"")
        print("   or: python main.py --interactive")
        print("Example: python main.py \"What does the local documentation say about RAG?\"")
        sys.exit(0)

    if not args.interactive:
        query = args.query.strip()
        if not query:
            print("Error: Query cannot be empty or whitespace.")
            sys.exit(1)

    load_dotenv()
    
    # 1. Initialize RAG
    try:
        initialize_knowledge_base()
    except Exception as e:
        print(f"Error initializing local knowledge base: {e}")
        # Proceeding anyway as they might not need local knowledge for general queries.
        
    if args.interactive:
        from memory import create_session
        session = create_session()
        print("\n" + "=" * 40)
        print("AI RESEARCH AGENT (Interactive Mode)")
        print("Type 'exit' or 'quit' to terminate.")
        print("=" * 40 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                final_answer, state = execute_agent(user_input, session=session)
                print(f"\nAgent: {final_answer}\n")
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"\n[Error]: {e}\n")
        
        print("\nExiting interactive session.")
        sys.exit(0)
        
    print("\n" + "=" * 40)
    print("AI RESEARCH AGENT")
    print("=" * 40)
    print(f"\nQuery:\n{query}\n")
    print("-" * 40)
    print("EXECUTION")
    print("-" * 40)

    # We want to capture output nicely. `execute_agent` prints to stdout.
    # We will let it print normally, but if it fails, handle it gracefully.
    try:
        start_time = time.time()
        final_answer, state = execute_agent(query)
    except ValueError as e:
        print(f"\n[Agent Error]: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[Agent Runtime Error]: {e}")
        sys.exit(1)
    except errors.APIError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\nGemini API quota is currently exhausted.")
            print("Please try again after the quota resets.")
            sys.exit(1)
        else:
            print(f"\n[Gemini API Error]: {e}")
            sys.exit(1)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\nGemini API quota is currently exhausted.")
            print("Please try again after the quota resets.")
            sys.exit(1)
        print(f"\n[Unexpected Error]: {e}")
        sys.exit(1)

    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Query: {query}")
    print(f"Iterations: {state.iteration}")
    print(f"Tool calls: {len(state.tool_calls)}")
    print(f"Retrieved chunks: {len(state.retrieved_evidence)}")
    print(f"Reflection attempts: {state.reflection_attempts}")
    print(f"Duration: {duration:.2f}s")
    
    print("\n" + "-" * 40)
    print("FINAL ANSWER")
    print("-" * 40 + "\n")
    print(final_answer)

    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
