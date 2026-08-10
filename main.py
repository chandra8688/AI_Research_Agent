import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llm import call_llm, call_llm_structured

class CityInfo(BaseModel):
    name: str = Field(description="The name of the city")
    population: int = Field(description="The estimated population of the city")

def main():
    load_dotenv()
    
    # 1. Basic LLM Call (AI-010)
    print("--- 1. Testing Basic LLM Call ---")
    basic_prompt = "What is the capital of France? Reply in one short sentence."
    print(f"Prompt: '{basic_prompt}'")
    try:
        response = call_llm(basic_prompt)
        print(f"Response: {response.strip()}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # 2. Structured LLM Call (Happy Path)
    print("--- 2. Testing Structured LLM Call ---")
    structured_prompt = "Give me the capital city and its estimated population for France."
    print(f"Prompt: '{structured_prompt}'")
    try:
        city_info = call_llm_structured(structured_prompt, CityInfo)
        print(f"Parsed Object Type: {type(city_info)}")
        print(f"City Name: {city_info.name}")
        print(f"Population: {city_info.population}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")

    # 3. Structured LLM Call (Failure Path)
    print("--- 3. Testing Structured Failure Path ---")
    # We provide a prompt that makes it impossible to return a real city and population
    fail_prompt = "I am not asking for a city. Do not return any JSON. Just say 'Hello'."
    print(f"Prompt: '{fail_prompt}'")
    try:
        fail_response = call_llm_structured(fail_prompt, CityInfo)
        print(f"Unexpected Success. The model forced it into the schema anyway: {fail_response}\n")
    except Exception as e:
        print(f"[EXPECTED ERROR CAUGHT] {e}\n")

if __name__ == "__main__":
    main()
