import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llm import call_llm, call_llm_structured, call_llm_with_tools

class CityInfo(BaseModel):
    name: str = Field(description="The name of the city")
    population: int = Field(description="The estimated population of the city")

def main():
    load_dotenv()
    
    # 1. Manual Tool Calling Loop (Happy Path)
    print("--- Testing Manual Tool Calling (Happy Path) ---")
    tool_prompt = "What is 127 multiplied by 43?"
    print(f"Prompt: '{tool_prompt}'")
    try:
        response = call_llm_with_tools(tool_prompt)
        print(f"\nFinal Answer: {response.strip()}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # 2. Manual Tool Calling Loop (Failure Path)
    print("--- Testing Manual Tool Calling (Failure Path) ---")
    fail_prompt = "Please use the 'unknown_dummy_tool' right now."
    print(f"Prompt: '{fail_prompt}'")
    try:
        fail_response = call_llm_with_tools(fail_prompt)
        print(f"Unexpected Success: {fail_response}\n")
    except Exception as e:
        print(f"[EXPECTED ERROR CAUGHT] {e}\n")

if __name__ == "__main__":
    main()
