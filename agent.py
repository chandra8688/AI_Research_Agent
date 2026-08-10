import os
from google import genai
from google.genai import errors
from google.genai import types
from tools import calculate_product, search_web, search_local_knowledge

# ---------------------------------------------------------------------------
# Tool Registry: maps function name → local Python callable
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "calculate_product": calculate_product,
    "search_web": search_web,
    "search_local_knowledge": search_local_knowledge,
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
    types.FunctionDeclaration(
        name="search_web",
        description=(
            "Searches the web for information on a given query using DuckDuckGo. "
            "Returns a list of results with title, URL, and a short snippet. "
            "Use this when you need current or specific external information to answer a question."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The search query string"},
                "max_results": {
                    "type": "INTEGER",
                    "description": "Maximum number of results to return (1-10, default 3)",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="search_local_knowledge",
        description=(
            "Searches the local document knowledge base for internal information on a given query. "
            "Use this ONLY when the user asks about local, internal, or provided documents, or topics "
            "where you need context from the internal knowledge base to answer properly."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The search query string"},
            },
            "required": ["query"],
        },
    ),
]

MODEL = "gemini-3.5-flash"
MAX_ITERATIONS = 5


from state import AgentState

def execute_agent(prompt: str, max_iterations: int = MAX_ITERATIONS) -> tuple[str, AgentState]:
    """
    Runs a ReAct-style agent loop and returns both the final answer and the execution state.
    """
    # 1. INPUT VALIDATION
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Agent prompt must be a non-empty string.")
    prompt = prompt.strip()

    # 5. ITERATION GUARD (input validation)
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be greater than 0.")

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
    
    # Initialize explicit agent state
    state = AgentState(query=prompt, contents=contents)

    for iteration in range(1, max_iterations + 1):
        print(f"\n[AGENT ITERATION {iteration}]")
        state.iteration = iteration

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=state.contents,
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
            state.contents.append(response.candidates[0].content)

            # Route to registered tool (2. TOOL NAME VALIDATION)
            tool_fn = TOOL_REGISTRY.get(fc.name)
            if tool_fn is None:
                raise RuntimeError(
                    f"Unknown tool requested: '{fc.name}'. "
                    "Only functions explicitly present in TOOL_REGISTRY may execute."
                )

            # Extract arguments
            kwargs = dict(fc.args)
            args_display = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            print(f"[TOOL CALL] {fc.name}({args_display})")
            
            # Record tool call in state
            state.tool_calls.append({"name": fc.name, "args": kwargs})

            # 3 & 4. TOOL ARGUMENT VALIDATION & EXECUTION ERROR HANDLING
            try:
                # Dispatch: cast args to the right types per tool
                if fc.name == "calculate_product":
                    typed_kwargs = {k: int(v) for k, v in kwargs.items()}
                elif fc.name == "search_web":
                    typed_kwargs = {
                        "query": str(kwargs["query"]),
                        **({"max_results": int(kwargs["max_results"])} if "max_results" in kwargs else {}),
                    }
                else:
                    typed_kwargs = kwargs  # fallback: pass raw strings

                result = tool_fn(**typed_kwargs)
            except Exception as e:
                # Controlled error instead of crashing
                result = f"Error: Tool execution failed: {str(e)}"
                print(f"[TOOL ERROR] {result}")

            print(f"[TOOL RESULT] {str(result)[:300]}{'...' if len(str(result)) > 300 else ''}")
            
            # Record tool result in state (8. STATE ERROR RECORDING)
            state.tool_results.append({"name": fc.name, "result": result})
            
            # Reflection processing
            if fc.name == "search_local_knowledge" and not str(result).startswith("Error:"):
                state.retrieved_evidence.append(result)
                state.reflection_attempts += 1
                
                # Evaluate evidence
                from reflection import evaluate_evidence
                reflection = evaluate_evidence(prompt, state.retrieved_evidence)
                state.reflection_result = reflection
                print(f"[REFLECTION] sufficient={reflection.sufficient}, reason={reflection.reason}")
                
                # 6. REFLECTION GUARD
                if not reflection.sufficient:
                    if state.reflection_attempts < 2:
                        result = (f"{result}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                                  f"because: {reflection.reason}. Please refine your search query and call search_local_knowledge again.")
                    else:
                        result = (f"{result}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                                  f"because: {reflection.reason}. MAX RETRIEVAL ATTEMPTS REACHED. "
                                  f"Please provide your final answer based only on what you have, or admit you do not know.")

            # Build and append the function response turn
            func_response_part = types.Part.from_function_response(
                name=fc.name,
                response={"result": result},
            )
            state.contents.append(
                types.Content(role="user", parts=[func_response_part])
            )
            # Continue to next iteration

        # -------------------------------------------------------------------
        # Branch B: LLM returned a final text answer → stop
        # -------------------------------------------------------------------
        else:
            print("[FINAL ANSWER]")
            final_text = response.text
            
            # 7. FINAL OUTPUT VALIDATION
            if not isinstance(final_text, str) or not final_text.strip():
                raise RuntimeError("Agent produced an empty final answer.")
                
            state.final_answer = final_text
            return final_text, state

    # 5. ITERATION GUARD (limit reached)
    raise RuntimeError(
        f"Agent did not produce a final answer within {max_iterations} iterations."
    )


def run_agent(prompt: str, max_iterations: int = MAX_ITERATIONS) -> str:
    """
    Runs a ReAct-style agent loop.
    Returns only the final string answer to preserve existing behavior.
    """
    answer, _ = execute_agent(prompt, max_iterations)
    return answer
