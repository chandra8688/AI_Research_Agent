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

from state import AgentState
from memory import AgentSession, create_session
from config import settings

def execute_agent(prompt: str, max_iterations: int | None = None, session: AgentSession | None = None) -> tuple[str, AgentState]:
    """
    Runs a ReAct-style agent loop and returns both the final answer and the execution state.
    """
    # 1. INPUT VALIDATION
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Agent prompt must be a non-empty string.")
    prompt = prompt.strip()

    if max_iterations is None:
        max_iterations = settings.max_agent_iterations

    # 5. ITERATION GUARD (input validation)
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be greater than 0.")

    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    client = genai.Client(api_key=api_key)

    # Build the tool configuration once
    tool_config = types.Tool(function_declarations=FUNCTION_DECLARATIONS)
    config = types.GenerateContentConfig(tools=[tool_config])

    from planning import create_research_plan
    plan = create_research_plan(prompt)

    if session is None:
        session = create_session()

    # Add the current user prompt to the session memory
    session.memory.add_user_message(prompt)

    # Initialise conversation history with the bounded session memory
    # We only take the last 10 messages to avoid blowing up the context window
    MAX_CONTEXT_MESSAGES = 10
    recent_messages = session.memory.get_messages()[-MAX_CONTEXT_MESSAGES:]
    
    contents = []
    for msg in recent_messages:
        contents.append(
            types.Content(role=msg.role, parts=[types.Part.from_text(text=msg.content)])
        )
        
    # Inject research plan guidance
    guidance = "\n\n[SYSTEM GUIDANCE based on intent classification]:\n"
    if plan.requires_local_knowledge:
        guidance += "- Local knowledge should be considered. Consider using search_local_knowledge.\n"
    if plan.requires_web:
        guidance += "- Web search should be considered. Consider using search_web.\n"
    if plan.requires_calculation:
        guidance += "- Calculation is required. Consider using calculate_product.\n"
    if not plan.requires_local_knowledge and not plan.requires_web and not plan.requires_calculation:
        guidance += "- No specialized tools required. Allow direct answering.\n"
        
    # Append guidance to the last user message in contents (which is the current prompt)
    if contents and contents[-1].role == "user":
        contents[-1].parts[0].text += guidance
    
    # Initialize explicit agent state
    state = AgentState(query=prompt, contents=contents)
    state.research_plan = plan
    
    plan_details = {
        "intent": plan.intent,
        "requires_web": plan.requires_web,
        "requires_local_knowledge": plan.requires_local_knowledge,
        "requires_calculation": plan.requires_calculation,
        "requires_multi_source_research": plan.requires_multi_source_research,
        "step_count": len(plan.steps)
    }
    state.add_trace("research_plan", plan_details)
    state.add_trace("agent_start", {"query": prompt})

    for iteration in range(1, max_iterations + 1):
        print(f"\n[AGENT ITERATION {iteration}]")
        state.iteration = iteration
        state.add_trace("iteration_start")

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=state.contents,
                config=config,
            )
        except errors.APIError as e:
            state.add_trace("agent_error", {"error": f"Gemini API Error: {str(e)}"})
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
            state.add_trace("tool_call", {"tool_name": fc.name, "arguments": kwargs})

            if fc.name in ["search_local_knowledge", "search_web"]:
                state.add_trace("research_source_selected", {"source": fc.name})

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
            result_str = str(result)
            state.add_trace("tool_result", {"result_preview": result_str[:300] + ('...' if len(result_str) > 300 else '')})
            
            # Evidence collection for multi-source
            from research import parse_local_evidence, parse_web_evidence, format_combined_evidence
            needs_reflection = False
            
            if fc.name == "search_local_knowledge" and not str(result).startswith("Error:"):
                items = parse_local_evidence(str(result))
                state.multi_source_evidence.extend(items)
                state.retrieved_evidence.append(result) # Preserve old behavior
                needs_reflection = True
                
            elif fc.name == "search_web" and not str(result).startswith("Error:"):
                items = parse_web_evidence(str(result))
                state.multi_source_evidence.extend(items)
                needs_reflection = True
                
            if needs_reflection:
                state.add_trace("evidence_collected", {"count": len(state.multi_source_evidence)})
                
                # If plan requires multi-source, ensure both are collected
                if plan.requires_multi_source_research:
                    has_local = any(e.source_type == "local" for e in state.multi_source_evidence)
                    has_web = any(e.source_type == "web" for e in state.multi_source_evidence)
                    if not (has_local and has_web):
                        # Still missing one source
                        result = (f"{result}\n\n[SYSTEM GUIDANCE]: You have collected partial evidence. "
                                  f"Your research plan requires BOTH local and web sources. "
                                  f"Please call the other required search tool.")
                        func_response_part = types.Part.from_function_response(name=fc.name, response={"result": result})
                        state.contents.append(types.Content(role="user", parts=[func_response_part]))
                        continue

                # We have all required evidence (or it's a single source plan). Synthesize and reflect.
                combined_text = format_combined_evidence(state.multi_source_evidence)
                state.add_trace("evidence_synthesis", {"sources": [e.source for e in state.multi_source_evidence]})
                
                state.reflection_attempts += 1
                from reflection import evaluate_evidence
                # Reflect on the combined evidence text
                reflection = evaluate_evidence(prompt, [combined_text])
                state.reflection_result = reflection
                print(f"[REFLECTION] sufficient={reflection.sufficient}, reason={reflection.reason}")
                state.add_trace("reflection", {"status": "sufficient" if reflection.sufficient else "insufficient", "feedback": reflection.reason})
                
                if not reflection.sufficient:
                    if state.reflection_attempts < settings.max_reflection_attempts:
                        result = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                                  f"because: {reflection.reason}. Please refine your search query.")
                    else:
                        result = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                                  f"because: {reflection.reason}. MAX RETRIEVAL ATTEMPTS REACHED. "
                                  f"Please provide your final answer based only on what you have, or admit you do not know.")
                else:
                    # Inject synthesis instructions
                    result = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence is SUFFICIENT. "
                              f"Please generate the final answer. "
                              f"\n\nSYNTHESIS INSTRUCTIONS:\n"
                              f"- Distinguish between local evidence and web evidence in your answer.\n"
                              f"- Do not invent unsupported claims.\n"
                              f"- Distinguish conflicting evidence.\n"
                              f"- Prefer explicit evidence over assumptions.\n"
                              f"- Identify the source type supporting important claims.\n"
                              f"- Add source attribution to the final answer context: [LOCAL: filename] or [WEB: URL/title].")

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
                state.add_trace("agent_error", {"error": "Agent produced an empty final answer."})
                raise RuntimeError("Agent produced an empty final answer.")
                
            state.final_answer = final_text
            state.add_trace("final_answer")
            
            # Save assistant response to session memory
            session.memory.add_assistant_message(final_text)
            
            return final_text, state

    # 5. ITERATION GUARD (limit reached)
    err_msg = f"Agent did not produce a final answer within {max_iterations} iterations."
    state.add_trace("agent_error", {"error": err_msg})
    raise RuntimeError(err_msg)


def run_agent(prompt: str, max_iterations: int | None = None, session: AgentSession | None = None) -> str:
    """
    Runs a ReAct-style agent loop.
    Returns only the final string answer to preserve existing behavior.
    """
    answer, _ = execute_agent(prompt, max_iterations, session)
    return answer
