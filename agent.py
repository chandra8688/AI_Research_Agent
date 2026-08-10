import os
from google import genai
from google.genai import errors
from google.genai import types
from tools import calculate_product

# ---------------------------------------------------------------------------
# Tool Registry: maps function name → local Python callable
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "calculate_product": calculate_product,
}

# ---------------------------------------------------------------------------
# Function Declarations: tells Gemini what tools are available and their shape
# ---------------------------------------------------------------------------
FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="calculate_product",
        description="Calculates the product of two integers. Use this whenever multiplication is needed.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "a": {"type": "INTEGER", "description": "The first integer"},
                "b": {"type": "INTEGER", "description": "The second integer"},
            },
            "required": ["a", "b"],
        },
    ),
]

MODEL = "gemini-3.5-flash"
MAX_ITERATIONS = 5


def run_agent(prompt: str, max_iterations: int = MAX_ITERATIONS) -> str:
    """
    Runs a ReAct-style agent loop.

    Each iteration:
      1. Sends current conversation history to Gemini.
      2. If Gemini requests a tool → execute it locally, append the result, continue.
      3. If Gemini returns text → stop and return the final answer.

    Raises RuntimeError if max_iterations is reached without a final answer.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    client = genai.Client(api_key=api_key)

    # Build the tool configuration once
    tool_config = types.Tool(function_declarations=FUNCTION_DECLARATIONS)
    config = types.GenerateContentConfig(tools=[tool_config])

    # Initialise conversation history with the user prompt
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n[AGENT ITERATION {iteration}]")

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
        except errors.APIError as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        # -------------------------------------------------------------------
        # Branch A: LLM requested a tool call
        # -------------------------------------------------------------------
        if response.function_calls:
            fc = response.function_calls[0]

            # Append the model's function-call turn to history
            contents.append(response.candidates[0].content)

            # Route to registered tool
            tool_fn = TOOL_REGISTRY.get(fc.name)
            if tool_fn is None:
                raise RuntimeError(
                    f"LLM requested unknown tool '{fc.name}'. "
                    "Add it to TOOL_REGISTRY in agent.py."
                )

            # Extract arguments and execute locally
            kwargs = dict(fc.args)
            args_display = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[TOOL CALL] {fc.name}({args_display})")

            result = tool_fn(**{k: int(v) for k, v in kwargs.items()})
            print(f"[TOOL RESULT] {result}")

            # Build and append the function response turn
            func_response_part = types.Part.from_function_response(
                name=fc.name,
                response={"result": result},
            )
            contents.append(
                types.Content(role="user", parts=[func_response_part])
            )
            # Continue to next iteration

        # -------------------------------------------------------------------
        # Branch B: LLM returned a final text answer → stop
        # -------------------------------------------------------------------
        else:
            print("[FINAL ANSWER]")
            return response.text

    # If we exit the loop without a final answer, guard kicked in
    raise RuntimeError(
        f"Agent did not produce a final answer within {max_iterations} iterations."
    )
