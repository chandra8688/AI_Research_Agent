import logging
import re
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from config import settings
from planning import create_research_plan
from state import AgentState
from memory import create_session, AgentSession
from agent import TOOL_REGISTRY, FUNCTION_DECLARATIONS
from research import parse_local_evidence, parse_web_evidence, format_combined_evidence
from reflection import evaluate_evidence
from quality import extract_claims, assess_claims, validate_citations

logger = logging.getLogger(__name__)

# Number of prior conversation messages forwarded to the model as context.
MAX_CONTEXT_MESSAGES = 10

_PROVIDER_ERROR_NAMES = ("RetryableProviderError", "FatalProviderError", "ProviderError")


class GraphState(TypedDict):
    prompt: str
    max_iterations: int
    session: Any
    agent_state: AgentState
    llm_response: Any
    error: str | None
    next_action: str
    is_simple: bool


def _is_provider_error(e: Exception) -> bool:
    from providers.errors import ProviderError
    return isinstance(e, ProviderError) or type(e).__name__ in _PROVIDER_ERROR_NAMES


_FULLWIDTH_CITATION = re.compile(r"【\s*(WEB|LOCAL)\s*:\s*([^】]*)】")


def normalize_citation_brackets(text: str) -> str:
    """Some models (e.g. gpt-oss via Groq) write citations as 【WEB: ...】. Convert them
    to the [WEB: ...] form so citation validation and the UI recognise them."""
    return _FULLWIDTH_CITATION.sub(lambda m: f"[{m.group(1)}: {m.group(2).strip()}]", text)


def _get_primary_provider():
    from providers import get_provider
    return get_provider(settings.llm_primary_provider or settings.llm_provider)


def _function_response_message(pending: list[dict], last_result_override: str | None = None) -> dict:
    """Builds ONE user message containing a function response for EVERY pending tool call.

    Providers require one function response per function call in the preceding model
    turn (Gemini rejects mismatched counts; OpenAI-style APIs require a tool message per
    tool_call_id). If `last_result_override` is given, it replaces the result of the last
    call (used to attach reflection feedback to the evidence).
    """
    parts = []
    for i, call in enumerate(pending):
        result = call["result"]
        if last_result_override is not None and i == len(pending) - 1:
            result = last_result_override
        part = {"name": call["name"], "response": {"result": result}}
        if call.get("id"):
            part["id"] = call["id"]
        parts.append(part)
    return {"role": "user", "function_responses": parts}


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

    if session is None:
        session = create_session()

    session.memory.add_user_message(prompt)

    recent_messages = session.memory.get_messages()[-MAX_CONTEXT_MESSAGES:]
    contents = [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    agent_state = AgentState(query=prompt, contents=contents)
    agent_state.add_trace("graph_start")

    from planning import is_simple_query
    is_simple = is_simple_query(prompt)
    if is_simple:
        agent_state.add_trace("classified_as_simple_query")

    return {
        "prompt": prompt,
        "max_iterations": max_iterations,
        "session": session,
        "agent_state": agent_state,
        "error": None,
        "is_simple": is_simple
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
    guidance += ("- When you cite retrieved evidence, use [LOCAL: filename] or "
                 "[WEB: Page Title (URL)] exactly as the sources appear.\n")

    if agent_state.contents and agent_state.contents[-1].get("role") == "user":
        agent_state.contents[-1]["content"] += guidance

    return {"agent_state": agent_state}


def agent_decide(state: GraphState):
    agent_state = state["agent_state"]
    max_iterations = state["max_iterations"]

    if agent_state.iteration >= max_iterations:
        if agent_state.multi_source_evidence:
            # Evidence was gathered but the model kept calling tools. Rather than
            # discarding the research, synthesise an answer from what we have.
            agent_state.add_trace("iteration_limit_reached", {"max_iterations": max_iterations})
            return {"next_action": "force_synthesis"}
        err_msg = f"Agent did not produce a final answer within {max_iterations} iterations."
        agent_state.add_trace("agent_error", {"error": err_msg})
        return {"error": err_msg, "next_action": "end"}

    agent_state.iteration += 1
    agent_state.add_trace("iteration_start")
    logger.info("Agent iteration %d", agent_state.iteration)

    provider = _get_primary_provider()

    try:
        response = provider.generate_agent_step(agent_state.contents, FUNCTION_DECLARATIONS)
    except Exception as e:
        if _is_provider_error(e):
            agent_state.add_trace("agent_error", {"error": str(e)})
            return {"error": e, "next_action": "end"}
        err_msg = f"Provider API Error: {str(e)}"
        agent_state.add_trace("agent_error", {"error": err_msg})
        return {"error": err_msg, "next_action": "end"}

    if response.function_calls:
        agent_state.contents.append(response.model_message)
        return {"llm_response": response, "next_action": "tools", "agent_state": agent_state}
    return {"llm_response": response, "next_action": "quality_check", "agent_state": agent_state}


def _coerce_tool_args(name: str, kwargs: dict) -> dict:
    if name == "calculate_product":
        return {k: int(v) for k, v in kwargs.items()}
    if name == "search_web":
        return {
            "query": str(kwargs["query"]),
            **({"max_results": int(kwargs["max_results"])} if "max_results" in kwargs else {}),
        }
    return kwargs


def execute_tools(state: GraphState):
    """Executes EVERY function call the model requested in this turn.

    Models (notably Gemini) may request several tools in parallel. Each call must get its
    own function response in the next turn, so all of them are executed and recorded.
    """
    agent_state = state["agent_state"]
    response = state["llm_response"]

    pending = []
    for fc in response.function_calls:
        kwargs = dict(fc.args or {})
        agent_state.tool_calls.append({"name": fc.name, "args": kwargs})
        agent_state.add_trace("tool_call", {"tool_name": fc.name, "arguments": kwargs})

        tool_fn = TOOL_REGISTRY.get(fc.name)
        if tool_fn is None:
            result = f"Error: Unknown tool requested: '{fc.name}'."
            agent_state.add_trace("agent_error", {"error": result})
        else:
            if fc.name in ["search_local_knowledge", "search_web"]:
                agent_state.add_trace("research_source_selected", {"source": fc.name})
            try:
                result = tool_fn(**_coerce_tool_args(fc.name, kwargs))
            except Exception as e:
                result = f"Error: Tool execution failed: {str(e)}"

        agent_state.tool_results.append({"name": fc.name, "result": result})
        result_str = str(result)
        agent_state.add_trace("tool_result", {
            "tool_name": fc.name,
            "result_preview": result_str[:300] + ('...' if len(result_str) > 300 else '')
        })
        pending.append({"name": fc.name, "id": getattr(fc, "id", None), "result": result})

    agent_state.pending_tool_calls = pending
    return {"agent_state": agent_state, "next_action": "collect_evidence"}


def collect_evidence(state: GraphState):
    agent_state = state["agent_state"]
    pending = agent_state.pending_tool_calls
    plan = agent_state.research_plan

    needs_reflection = False
    for call in pending:
        result_str = str(call["result"])
        if result_str.startswith("Error:"):
            continue
        if call["name"] == "search_local_knowledge":
            items = parse_local_evidence(result_str)
            agent_state.multi_source_evidence.extend(items)
            agent_state.retrieved_evidence.append(result_str)
            needs_reflection = True
        elif call["name"] == "search_web":
            items = parse_web_evidence(result_str)
            agent_state.multi_source_evidence.extend(items)
            needs_reflection = True

    if needs_reflection:
        agent_state.add_trace("evidence_collected", {"count": len(agent_state.multi_source_evidence)})

        if plan is not None and plan.requires_multi_source_research:
            has_local = any(e.source_type == "local" for e in agent_state.multi_source_evidence)
            has_web = any(e.source_type == "web" for e in agent_state.multi_source_evidence)
            if not (has_local and has_web):
                # Only one channel collected so far: hand the raw results back and let the
                # agent query the other channel before reflecting.
                agent_state.contents.append(_function_response_message(pending))
                agent_state.pending_tool_calls = []
                return {"agent_state": agent_state, "next_action": "agent_decide"}

        return {"agent_state": agent_state, "next_action": "reflection"}

    agent_state.contents.append(_function_response_message(pending))
    agent_state.pending_tool_calls = []
    return {"agent_state": agent_state, "next_action": "agent_decide"}


def reflection(state: GraphState):
    agent_state = state["agent_state"]
    prompt = state["prompt"]
    pending = agent_state.pending_tool_calls

    combined_text = format_combined_evidence(agent_state.multi_source_evidence)
    agent_state.add_trace("evidence_synthesis", {"sources": [e.source for e in agent_state.multi_source_evidence]})

    agent_state.reflection_attempts += 1
    try:
        reflection_obj = evaluate_evidence(prompt, [combined_text])
    except Exception as e:
        # The judge is advisory. If it fails (rate limit, malformed JSON), continue to
        # synthesis with the evidence we have instead of failing the whole request.
        from reflection import ReflectionResult
        logger.warning("Reflection failed, proceeding to synthesis: %s", e)
        agent_state.add_trace("reflection_error", {"error": str(e)})
        reflection_obj = ReflectionResult(sufficient=True, reason="Reflection unavailable; using collected evidence.")
    agent_state.reflection_result = reflection_obj
    agent_state.add_trace("reflection", {"status": "sufficient" if reflection_obj.sufficient else "insufficient", "feedback": reflection_obj.reason})

    if not reflection_obj.sufficient:
        # Limit reached: hard-route to force_synthesis — do NOT return to agent_decide.
        # The model cannot override this; routing is enforced at the graph level.
        if agent_state.reflection_attempts >= settings.max_reflection_attempts:
            agent_state.add_trace("reflection_limit_reached", {
                "reflection_attempts": agent_state.reflection_attempts,
                "max_reflection_attempts": settings.max_reflection_attempts
            })
            agent_state.pending_tool_calls = []
            return {"agent_state": agent_state, "next_action": "force_synthesis"}

        # Attempts remain: inject INSUFFICIENT signal and allow another research iteration.
        res_mod = (f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence was evaluated as INSUFFICIENT "
                   f"because: {reflection_obj.reason}. Please refine your search query.")
    else:
        # Evidence sufficient: inject SUFFICIENT signal and allow agent to generate final answer.
        res_mod = (
            f"{combined_text}\n\n[SYSTEM EVALUATION]: The retrieved evidence is SUFFICIENT. "
            f"Please generate the final answer. "
            f"\n\nSYNTHESIS INSTRUCTIONS:\n"
            f"- Distinguish between local evidence and web evidence in your answer.\n"
            f"- Do not invent unsupported claims.\n"
            f"- Distinguish conflicting evidence.\n"
            f"- Prefer explicit evidence over assumptions.\n"
            f"- Identify the source type supporting important claims.\n"
            f"- Cite sources using exactly the identifiers shown above: [LOCAL: filename] or [WEB: Page Title (URL)]."
        )

    if pending:
        agent_state.contents.append(_function_response_message(pending, last_result_override=res_mod))
    else:
        agent_state.contents.append({"role": "user", "content": res_mod})
    agent_state.pending_tool_calls = []

    return {"agent_state": agent_state, "next_action": "agent_decide"}


def force_synthesis(state: GraphState):
    """Called when max_reflection_attempts (or the iteration limit) is reached.
    Bypasses agent_decide entirely — the model is NOT offered tools here.
    Makes a direct generate() call with all accumulated evidence and a strict synthesis prompt.
    Routes to quality_check for the standard grounding/citation pipeline.
    """
    agent_state = state["agent_state"]
    prompt = state["prompt"]

    combined_text = format_combined_evidence(agent_state.multi_source_evidence)

    synthesis_prompt = (
        f"You are a research assistant. Based ONLY on the following evidence, "
        f"provide a comprehensive answer to the query.\n\n"
        f"QUERY:\n{prompt}\n\n"
        f"EVIDENCE:\n{combined_text}\n\n"
        f"SYNTHESIS INSTRUCTIONS:\n"
        f"- Answer only from the provided evidence. Do not invent unsupported claims.\n"
        f"- Distinguish demonstrated results from announced targets.\n"
        f"- Every factual claim based on retrieved evidence should include a citation.\n"
        f"- Use ONLY exact retrieved source identifiers for citations.\n"
        f"- Web citation format: [WEB: title (URL)]\n"
        f"- Local citation format: [LOCAL: filename]\n"
        f"- Never invent or modify source identifiers.\n\n"
        f"PROVIDE YOUR FINAL ANSWER NOW:"
    )

    from providers import AgentResponse
    provider = _get_primary_provider()

    try:
        final_text = provider.generate(synthesis_prompt)
    except Exception as e:
        if _is_provider_error(e):
            agent_state.add_trace("agent_error", {"error": str(e)})
            return {"agent_state": agent_state, "error": e, "next_action": "end"}
        final_text = f"Evidence was gathered but synthesis failed: {str(e)}"

    if not final_text or not final_text.strip():
        final_text = (
            "The research agent gathered evidence but the model returned an empty synthesis. "
            "Please try a more specific question."
        )

    # Wrap in AgentResponse so quality_check can process it via the standard pipeline.
    response = AgentResponse(
        text=final_text,
        function_calls=[],
        model_message={"role": "model", "content": final_text}
    )

    agent_state.add_trace("force_synthesis", {"evidence_sources": len(agent_state.multi_source_evidence)})

    return {"agent_state": agent_state, "llm_response": response, "next_action": "quality_check"}


def fast_llm_path(state: GraphState):
    """Directly answers simple queries bypassing the tool loop."""
    agent_state = state["agent_state"]
    prompt = state["prompt"]

    from providers import AgentResponse
    provider = _get_primary_provider()

    # Include prior turns so follow-ups like "who is he?" keep their context.
    history = agent_state.contents[:-1]
    if history:
        transcript = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in history if m.get("content")
        )
        llm_prompt = f"Conversation so far:\n{transcript}\n\nUser: {prompt}\nAssistant:"
    else:
        llm_prompt = prompt

    try:
        final_text = provider.generate(llm_prompt)
    except Exception as e:
        if _is_provider_error(e):
            agent_state.add_trace("agent_error", {"error": str(e)})
            return {"agent_state": agent_state, "error": e, "next_action": "end"}
        final_text = f"Error generating fast response: {str(e)}"

    if not final_text or not final_text.strip():
        final_text = "The model returned an empty response."

    response = AgentResponse(
        text=final_text,
        function_calls=[],
        model_message={"role": "model", "content": final_text}
    )

    agent_state.add_trace("fast_llm_path")
    return {"agent_state": agent_state, "llm_response": response, "next_action": "quality_check"}


def quality_check(state: GraphState):
    agent_state = state["agent_state"]
    response = state["llm_response"]
    session = state["session"]
    final_text = response.text

    if not isinstance(final_text, str) or not final_text.strip():
        agent_state.add_trace("agent_error", {"error": "Agent produced an empty final answer."})
        return {"error": "Agent produced an empty final answer.", "next_action": "end"}

    final_text = normalize_citation_brackets(final_text)

    if state.get("is_simple", False) or not agent_state.multi_source_evidence:
        # Grounding checks claims against retrieved evidence. With no evidence (fast path,
        # greetings, or the model answering from its own knowledge) every claim would score
        # as unsupported, triggering pointless revision and rewrite calls that strip valid
        # content. Record that the answer is unverified instead.
        if not state.get("is_simple", False):
            agent_state.add_trace("grounding_skipped", {"reason": "No evidence was retrieved; answer is from model knowledge."})
        agent_state.final_answer = final_text
        agent_state.add_trace("final_answer")
        agent_state.add_trace("graph_end")
        session.memory.add_assistant_message(final_text)
        return {"agent_state": agent_state, "next_action": "end"}

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
    citations_found = len(re.findall(r'\[(?:LOCAL|WEB):\s*(.*?)\]', final_text))
    agent_state.add_trace("citation_validation", {
        "evidence_items": len(agent_state.multi_source_evidence),
        "citations_found": citations_found,
        "invalid_citations": invalid_citations
    })

    if not agent_state.refinement_attempted and (quality_report.unsupported_claims or invalid_citations):
        agent_state.refinement_attempted = True

        warning = "\n\n[SYSTEM GUIDANCE: GROUNDING WARNING]\n"
        if quality_report.unsupported_claims:
            warning += "Some claims are not supported by the collected evidence. Remove or qualify unsupported claims.\n"
        if invalid_citations:
            warning += "Some citations are invalid or do not match retrieved sources. Only cite retrieved sources exactly.\n"
        warning += "Please revise your answer. This is your only refinement attempt."

        agent_state.add_trace("refinement_requested", {
            "unsupported_claims": len(quality_report.unsupported_claims),
            "invalid_citations": len(invalid_citations)
        })
        agent_state.contents.append(response.model_message)
        agent_state.contents.append({"role": "user", "content": warning})

        return {"agent_state": agent_state, "next_action": "agent_decide"}

    if quality_report.overall_grounding_score < settings.grounding_threshold or quality_report.unsupported_claims or quality_report.conflicts_detected:
        from quality import apply_grounding_gate
        agent_state.add_trace("grounding_gate_triggered", {
            "score": quality_report.overall_grounding_score,
            "unsupported": len(quality_report.unsupported_claims)
        })
        logger.debug("Pre-gate answer:\n%s", final_text)
        final_text = apply_grounding_gate(final_text, quality_report, agent_state.multi_source_evidence)
        logger.debug("Post-gate answer:\n%s", final_text)

    agent_state.final_answer = final_text
    agent_state.add_trace("final_answer")
    agent_state.add_trace("graph_end")

    session.memory.add_assistant_message(final_text)
    return {"agent_state": agent_state, "next_action": "end"}


def _route_next(state: GraphState):
    if state.get("error"):
        return "end"
    return state.get("next_action")


route_agent_decide = _route_next
route_tools = _route_next
route_quality = _route_next
route_fast_path = _route_next


def route_validate(state: GraphState):
    if state.get("is_simple"):
        return "fast_llm_path"
    return "plan_research"


def route_collect(state: GraphState):
    return state.get("next_action")


def route_reflection(state: GraphState):
    """Routes reflection output: either back to agent_decide for more research,
    or to force_synthesis when the reflection attempt limit has been reached."""
    return state.get("next_action", "agent_decide")


def route_force_synthesis(state: GraphState):
    if state.get("error"):
        return "end"
    return state.get("next_action", "quality_check")


workflow = StateGraph(GraphState)
workflow.add_node("validate", validate)
workflow.add_node("plan_research", plan_research)
workflow.add_node("agent_decide", agent_decide)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("collect_evidence", collect_evidence)
workflow.add_node("reflection", reflection)
workflow.add_node("force_synthesis", force_synthesis)
workflow.add_node("fast_llm_path", fast_llm_path)
workflow.add_node("quality_check", quality_check)

workflow.add_edge(START, "validate")

workflow.add_conditional_edges(
    "validate",
    route_validate,
    {
        "fast_llm_path": "fast_llm_path",
        "plan_research": "plan_research"
    }
)

workflow.add_edge("plan_research", "agent_decide")

workflow.add_conditional_edges(
    "agent_decide",
    route_agent_decide,
    {
        "tools": "execute_tools",
        "quality_check": "quality_check",
        "force_synthesis": "force_synthesis",
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

# Reflection uses a conditional edge:
# - insufficient + attempts remain  → agent_decide (allow more research)
# - insufficient + limit reached    → force_synthesis (enforced, model cannot override)
# - sufficient                      → agent_decide (model will synthesize the answer)
workflow.add_conditional_edges(
    "reflection",
    route_reflection,
    {
        "agent_decide": "agent_decide",
        "force_synthesis": "force_synthesis"
    }
)

workflow.add_conditional_edges(
    "force_synthesis",
    route_force_synthesis,
    {
        "quality_check": "quality_check",
        "end": END
    }
)

workflow.add_conditional_edges(
    "fast_llm_path",
    route_fast_path,
    {
        "quality_check": "quality_check",
        "end": END
    }
)

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
    # Each tool-calling iteration visits several nodes; give LangGraph enough headroom
    # so our own iteration limits (not LangGraph's default of 25) decide when to stop.
    effective_iterations = max_iterations if isinstance(max_iterations, int) and max_iterations > 0 else settings.max_agent_iterations
    recursion_limit = max(50, effective_iterations * 8 + 20)
    final_state = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})

    agent_state = final_state.get("agent_state")
    if final_state.get("error"):
        err = final_state["error"]
        if isinstance(err, Exception):
            raise err
        raise RuntimeError(err)

    return agent_state.final_answer, agent_state
