import os
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from google import genai
from google.genai import errors, types
from config import settings
from planning import create_research_plan
from state import AgentState
from memory import create_session, AgentSession
from agent import TOOL_REGISTRY, FUNCTION_DECLARATIONS, MODEL
from research import parse_local_evidence, parse_web_evidence, format_combined_evidence
from reflection import evaluate_evidence
from quality import extract_claims, assess_claims, validate_citations

class GraphState(TypedDict):
    prompt: str
    max_iterations: int
    session: Any
    agent_state: AgentState
    llm_response: Any
    error: str | None
    next_action: str

def validate(state: GraphState):
    prompt = state.get("prompt", "")
    max_iterations = state.get("max_iterations")
    session = state.get("session")
    
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Agent prompt must be a non-empty string.")
    prompt = prompt.strip()
    
    if max_iterations is None:
        max_iterations = settings.max_agent_iterations
        
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be greater than 0.")
        
    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
        
    if session is None:
        session = create_session()
        
    session.memory.add_user_message(prompt)
    
    MAX_CONTEXT_MESSAGES = 10
    recent_messages = session.memory.get_messages()[-MAX_CONTEXT_MESSAGES:]
    
    contents = []
    for msg in recent_messages:
        contents.append(
            types.Content(role=msg.role, parts=[types.Part.from_text(text=msg.content)])
        )
        
    agent_state = AgentState(query=prompt, contents=contents)
    agent_state.add_trace("graph_start")
    
    return {
        "prompt": prompt,
        "max_iterations": max_iterations,
        "session": session,
        "agent_state": agent_state,
        "error": None
    }

def plan_research(state: GraphState):
    agent_state = state["agent_state"]
    prompt = state["prompt"]
    
    plan = create_research_plan(prompt)
    agent_state.research_plan = plan
    
    plan_details = {
        "intent": plan.intent,
        "requires_web": plan.requires_web,
        "requires_local_knowledge": plan.requires_local_knowledge,
        "requires_calculation": plan.requires_calculation,
        "requires_multi_source_research": plan.requires_multi_source_research,
        "step_count": len(plan.steps)
    }
    agent_state.add_trace("research_plan", plan_details)
    agent_state.add_trace("agent_start", {"query": prompt})
    
    guidance = "\n\n[SYSTEM GUIDANCE based on intent classification]:\n"
    if plan.requires_local_knowledge:
        guidance += "- Local knowledge should be considered. Consider using search_local_knowledge.\n"
    if plan.requires_web:
        guidance += "- Web search should be considered. Consider using search_web.\n"
    if plan.requires_calculation:
        guidance += "- Calculation is required. Consider using calculate_product.\n"
    if not plan.requires_local_knowledge and not plan.requires_web and not plan.requires_calculation:
        guidance += "- No specialized tools required. Allow direct answering.\n"
        
    if agent_state.contents and agent_state.contents[-1].role == "user":
        agent_state.contents[-1].parts[0].text += guidance
        
    return {"agent_state": agent_state}

def agent_decide(state: GraphState):
    agent_state = state["agent_state"]
    max_iterations = state["max_iterations"]
    
    if agent_state.iteration >= max_iterations:
        err_msg = f"Agent did not produce a final answer within {max_iterations} iterations."
        agent_state.add_trace("agent_error", {"error": err_msg})
        return {"error": err_msg, "next_action": "end"}
        
    agent_state.iteration += 1
    agent_state.add_trace("iteration_start")
    print(f"\n[AGENT ITERATION {agent_state.iteration}]")
    
    client = genai.Client(api_key=settings.gemini_api_key)
    tool_config = types.Tool(function_declarations=FUNCTION_DECLARATIONS)
    config = types.GenerateContentConfig(tools=[tool_config])
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=agent_state.contents,
            config=config,
        )
    except errors.APIError as e:
        err_msg = f"Gemini API Error: {str(e)}"
        agent_state.add_trace("agent_error", {"error": err_msg})
        return {"error": err_msg, "next_action": "end"}
        
    if response.function_calls:
        agent_state.contents.append(response.candidates[0].content)
        return {"llm_response": response, "next_action": "tools", "agent_state": agent_state}
    else:
        return {"llm_response": response, "next_action": "quality_check", "agent_state": agent_state}

def execute_tools(state: GraphState):
    agent_state = state["agent_state"]
    response = state["llm_response"]
    
    fc = response.function_calls[0]
    tool_fn = TOOL_REGISTRY.get(fc.name)
    if tool_fn is None:
        err = f"Unknown tool requested: '{fc.name}'."
        agent_state.add_trace("agent_error", {"error": err})
        return {"error": err, "next_action": "end"}
        
    kwargs = dict(fc.args)
    agent_state.tool_calls.append({"name": fc.name, "args": kwargs})
    agent_state.add_trace("tool_call", {"tool_name": fc.name, "arguments": kwargs})
    
    if fc.name in ["search_local_knowledge", "search_web"]:
        agent_state.add_trace("research_source_selected", {"source": fc.name})
        
    try:
        if fc.name == "calculate_product":
            typed_kwargs = {k: int(v) for k, v in kwargs.items()}
        elif fc.name == "search_web":
            typed_kwargs = {
                "query": str(kwargs["query"]),
                **({"max_results": int(kwargs["max_results"])} if "max_results" in kwargs else {}),
            }
        else:
            typed_kwargs = kwargs
            
        result = tool_fn(**typed_kwargs)
    except Exception as e:
        result = f"Error: Tool execution failed: {str(e)}"
        
    agent_state.tool_results.append({"name": fc.name, "result": result})
    result_str = str(result)
    agent_state.add_trace("tool_result", {"result_preview": result_str[:300] + ('...' if len(result_str) > 300 else '')})
    
    return {"agent_state": agent_state, "next_action": "collect_evidence"}

def collect_evidence(state: GraphState):
    agent_state = state["agent_state"]
    fc_name = agent_state.tool_calls[-1]["name"]
    result = agent_state.tool_results[-1]["result"]
    plan = agent_state.research_plan
    
    needs_reflection = False
    if fc_name == "search_local_knowledge" and not str(result).startswith("Error:"):
        items = parse_local_evidence(str(result))
        agent_state.multi_source_evidence.extend(items)
        agent_state.retrieved_evidence.append(result)
        needs_reflection = True
    elif fc_name == "search_web" and not str(result).startswith("Error:"):
        items = parse_web_evidence(str(result))
        agent_state.multi_source_evidence.extend(items)
        needs_reflection = True
        
    if needs_reflection:
        agent_state.add_trace("evidence_collected", {"count": len(agent_state.multi_source_evidence)})
        
        if plan.requires_multi_source_research:
            has_local = any(e.source_type == "local" for e in agent_state.multi_source_evidence)
            has_web = any(e.source_type == "web" for e in agent_state.multi_source_evidence)
            if not (has_local and has_web):
                res_mod = (f"{result}\n\n[SYSTEM GUIDANCE]: You have collected partial evidence. "
                          f"Your research plan requires BOTH local and web sources. "
                          f"Please call the other required search tool.")
                func_response_part = types.Part.from_function_response(name=fc_name, response={"result": res_mod})
                agent_state.contents.append(types.Content(role="user", parts=[func_response_part]))
                return {"agent_state": agent_state, "next_action": "agent_decide"}
                
        return {"agent_state": agent_state, "next_action": "reflection"}
        
    func_response_part = types.Part.from_function_response(name=fc_name, response={"result": result})
    agent_state.contents.append(types.Content(role="user", parts=[func_response_part]))
    return {"agent_state": agent_state, "next_action": "agent_decide"}

def reflection(state: GraphState):
    agent_state = state["agent_state"]
    prompt = state["prompt"]
    fc_name = agent_state.tool_calls[-1]["name"]
    
    combined_text = format_combined_evidence(agent_state.multi_source_evidence)
    agent_state.add_trace("evidence_synthesis", {"sources": [e.source for e in agent_state.multi_source_evidence]})
    
    agent_state.reflection_attempts += 1
    reflection_obj = evaluate_evidence(prompt, [combined_text])
    agent_state.reflection_result = reflection_obj
    agent_state.add_trace("reflection", {"status": "sufficient" if reflection_obj.sufficient else "insufficient", "feedback": reflection_obj.reason})
    
    if not reflection_obj.sufficient:
        if agent_state.reflection_attempts < settings.max_reflection_attempts:
            res_mod = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                      f"because: {reflection_obj.reason}. Please refine your search query.")
        else:
            res_mod = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                      f"because: {reflection_obj.reason}. MAX RETRIEVAL ATTEMPTS REACHED. "
                      f"Please provide your final answer based only on what you have, or admit you do not know.")
    else:
        res_mod = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence is SUFFICIENT. "
                  f"Please generate the final answer. "
                  f"\n\nSYNTHESIS INSTRUCTIONS:\n"
                  f"- Distinguish between local evidence and web evidence in your answer.\n"
                  f"- Do not invent unsupported claims.\n"
                  f"- Distinguish conflicting evidence.\n"
                  f"- Prefer explicit evidence over assumptions.\n"
                  f"- Identify the source type supporting important claims.\n"
                  f"- Add source attribution to the final answer context: [LOCAL: filename] or [WEB: URL/title].")
                  
    func_response_part = types.Part.from_function_response(name=fc_name, response={"result": res_mod})
    agent_state.contents.append(types.Content(role="user", parts=[func_response_part]))
    
    return {"agent_state": agent_state, "next_action": "agent_decide"}

def quality_check(state: GraphState):
    agent_state = state["agent_state"]
    response = state["llm_response"]
    session = state["session"]
    final_text = response.text
    
    if not isinstance(final_text, str) or not final_text.strip():
        agent_state.add_trace("agent_error", {"error": "Agent produced an empty final answer."})
        return {"error": "Agent produced an empty final answer.", "next_action": "end"}
        
    claims = extract_claims(final_text)
    agent_state.add_trace("claim_extraction", {"count": len(claims)})
    
    quality_report = assess_claims(claims, agent_state.multi_source_evidence)
    agent_state.research_quality = quality_report
    
    for assessment in quality_report.assessments:
        agent_state.add_trace("claim_assessment", {
            "claim": assessment.claim,
            "supported": assessment.supported,
            "reason": assessment.reason
        })
        
    agent_state.add_trace("grounding_check", {
        "claim_count": len(claims),
        "supported_claims": len(claims) - len(quality_report.unsupported_claims),
        "unsupported_claims": quality_report.unsupported_claims,
        "grounding_score": quality_report.overall_grounding_score,
        "conflicts_detected": quality_report.conflicts_detected
    })
    
    invalid_citations = validate_citations(final_text, agent_state.multi_source_evidence)
    agent_state.add_trace("citation_validation", {"invalid_citations": invalid_citations})
    
    refinement_attempted = getattr(agent_state, "refinement_attempted", False)
    if not refinement_attempted and (quality_report.unsupported_claims or invalid_citations):
        agent_state.refinement_attempted = True
        
        warning = "\n\n[SYSTEM GUIDANCE: GROUNDING WARNING]\n"
        if quality_report.unsupported_claims:
            warning += "Some claims are not supported by the collected evidence. Remove or qualify unsupported claims.\n"
        if invalid_citations:
            warning += "Some citations are invalid or do not match retrieved sources. Only cite retrieved sources exactly.\n"
        warning += "Please revise your answer. This is your only refinement attempt."
        
        agent_state.contents.append(response.candidates[0].content)
        agent_state.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=warning)]))
        
        return {"agent_state": agent_state, "next_action": "agent_decide"}
        
    agent_state.final_answer = final_text
    agent_state.add_trace("final_answer")
    agent_state.add_trace("graph_end")
    
    session.memory.add_assistant_message(final_text)
    return {"agent_state": agent_state, "next_action": "end"}

def route_agent_decide(state: GraphState):
    if state.get("error"):
        return "end"
    return state.get("next_action")

def route_tools(state: GraphState):
    if state.get("error"):
        return "end"
    return state.get("next_action")

def route_collect(state: GraphState):
    return state.get("next_action")

def route_quality(state: GraphState):
    if state.get("error"):
        return "end"
    return state.get("next_action")

workflow = StateGraph(GraphState)
workflow.add_node("validate", validate)
workflow.add_node("plan_research", plan_research)
workflow.add_node("agent_decide", agent_decide)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("collect_evidence", collect_evidence)
workflow.add_node("reflection", reflection)
workflow.add_node("quality_check", quality_check)

workflow.add_edge(START, "validate")
workflow.add_edge("validate", "plan_research")
workflow.add_edge("plan_research", "agent_decide")

workflow.add_conditional_edges(
    "agent_decide",
    route_agent_decide,
    {
        "tools": "execute_tools",
        "quality_check": "quality_check",
        "end": END
    }
)

workflow.add_conditional_edges(
    "execute_tools",
    route_tools,
    {
        "collect_evidence": "collect_evidence",
        "end": END
    }
)

workflow.add_conditional_edges(
    "collect_evidence",
    route_collect,
    {
        "reflection": "reflection",
        "agent_decide": "agent_decide"
    }
)

workflow.add_edge("reflection", "agent_decide")

workflow.add_conditional_edges(
    "quality_check",
    route_quality,
    {
        "agent_decide": "agent_decide",
        "end": END
    }
)

graph = workflow.compile()

def execute_agent_graph(prompt: str, max_iterations: int | None = None, session: AgentSession | None = None) -> tuple[str, AgentState]:
    initial_state = {
        "prompt": prompt,
        "max_iterations": max_iterations,
        "session": session,
        "error": None
    }
    final_state = graph.invoke(initial_state)
    
    agent_state = final_state.get("agent_state")
    if final_state.get("error"):
        raise RuntimeError(final_state["error"])
        
    return agent_state.final_answer, agent_state
