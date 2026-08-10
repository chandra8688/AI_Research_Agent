import os
from google import genai
from google.genai import errors
from google.genai import types
from pydantic import BaseModel
import tools

def call_llm(prompt: str) -> str:
    """Calls the Gemini API with the given prompt and returns the text response."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
    
    try:
        client = genai.Client(api_key=api_key)
        # Using gemini-3.5-flash as the default development model (2.5 is deprecated)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text
    except errors.APIError as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during LLM call: {str(e)}")

def call_llm_structured(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Calls the Gemini API and returns a structured response parsed into the given Pydantic schema."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return response.parsed
    except errors.APIError as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during structured LLM call: {str(e)}")

def call_llm_with_tools(prompt: str) -> str:
    """Manually handles a multi-turn tool calling loop with the Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
    
    client = genai.Client(api_key=api_key)
    
    # 1. Define the tool schema manually
    calc_func_decl = types.FunctionDeclaration(
        name="calculate_product",
        description="Calculates the product of two integers.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "a": {"type": "INTEGER", "description": "The first integer"},
                "b": {"type": "INTEGER", "description": "The second integer"}
            },
            "required": ["a", "b"]
        }
    )
    
    # We must also include a dummy unknown tool for our failure test
    dummy_func_decl = types.FunctionDeclaration(
        name="unknown_dummy_tool",
        description="A tool that doesn't exist locally, used to test failure handling.",
        parameters={"type": "OBJECT", "properties": {}}
    )
    
    tool_config = types.Tool(function_declarations=[calc_func_decl, dummy_func_decl])
    config = types.GenerateContentConfig(tools=[tool_config])
    
    # Setup conversation history
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=prompt)
        ])
    ]
    
    try:
        # First turn: Send prompt + tools
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=contents,
            config=config,
        )
        
        # 2. Inspect response for a function call
        if response.function_calls:
            fc = response.function_calls[0]
            
            # Append the model's response to the conversation history
            contents.append(response.candidates[0].content)
            
            # 3. Extract and Route
            if fc.name == "calculate_product":
                a = int(fc.args.get("a", 0))
                b = int(fc.args.get("b", 0))
                
                # 4. Execute locally & 5. Debug Print
                print(f"[TOOL EXECUTION] {fc.name}({a}, {b})")
                result = tools.calculate_product(a, b)
                
                # 6. Create function response
                func_response_part = types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result}
                )
                
                # Append the function response
                contents.append(
                    types.Content(role="user", parts=[func_response_part])
                )
                
                # 7. Send back to Gemini
                final_response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=contents,
                    config=config,
                )
                return final_response.text
            else:
                # Handle failure path: unregistered tool
                raise RuntimeError(f"Unknown tool requested by LLM: {fc.name}")
        else:
            # If no tools were called, just return the text
            return response.text
            
    except errors.APIError as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")
    except Exception as e:
        # Catch our RuntimeError and surface it
        if isinstance(e, RuntimeError):
            raise e
        raise RuntimeError(f"Unexpected error during tool loop: {str(e)}")
