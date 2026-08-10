from dataclasses import dataclass, field
from typing import Any, Dict
import time

@dataclass
class TraceEvent:
    timestamp: float
    event_type: str
    iteration: int
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    query: str
    contents: list = field(default_factory=list)
    iteration: int = 0
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    retrieved_evidence: list = field(default_factory=list)
    multi_source_evidence: list = field(default_factory=list)
    final_answer: str | None = None
    reflection_result: Any = None
    reflection_attempts: int = 0
    research_plan: Any = None
    research_quality: Any = None
    trace: list[TraceEvent] = field(default_factory=list)

    def add_trace(self, event_type: str, details: dict = None):
        if details is None:
            details = {}
        self.trace.append(TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            iteration=self.iteration,
            details=details
        ))

def format_trace(state: AgentState) -> str:
    lines = ["=" * 50, "AGENT TRACE", "=" * 50, ""]
    
    for idx, event in enumerate(state.trace, 1):
        if event.event_type == "research_plan":
            lines.append(f"[{idx}] RESEARCH PLAN")
            lines.append(f"Intent: {event.details.get('intent', '')}")
            lines.append(f"Requires web: {event.details.get('requires_web', False)}")
            lines.append(f"Requires local: {event.details.get('requires_local_knowledge', False)}")
            lines.append(f"Requires calculation: {event.details.get('requires_calculation', False)}")
            lines.append(f"Multi-source: {event.details.get('requires_multi_source_research', False)}")

        elif event.event_type == "agent_start":
            lines.append(f"[{idx}] AGENT START")
            lines.append(f"Query: {event.details.get('query', '')}")
            
        elif event.event_type == "iteration_start":
            lines.append(f"[{idx}] ITERATION {event.iteration}")
            
        elif event.event_type == "tool_call":
            lines.append(f"[{idx}] TOOL CALL")
            lines.append(f"Tool: {event.details.get('tool_name', '')}")
            if 'arguments' in event.details:
                lines.append(f"Arguments: {event.details['arguments']}")
                
        elif event.event_type == "tool_result":
            lines.append(f"[{idx}] TOOL RESULT")
            if 'result_preview' in event.details:
                lines.append(f"Result: {event.details['result_preview']}")
            if 'evidence_chunks' in event.details:
                lines.append(f"Evidence chunks: {event.details['evidence_chunks']}")

        elif event.event_type == "research_source_selected":
            lines.append(f"[{idx}] SOURCE SELECTED")
            lines.append(f"Source: {event.details.get('source', '')}")

        elif event.event_type == "evidence_collected":
            lines.append(f"[{idx}] EVIDENCE COLLECTED")
            lines.append(f"Count: {event.details.get('count', 0)}")

        elif event.event_type == "evidence_synthesis":
            lines.append(f"[{idx}] EVIDENCE SYNTHESIS")
            lines.append(f"Sources: {', '.join(event.details.get('sources', []))}")
                
        elif event.event_type == "reflection":
            lines.append(f"[{idx}] REFLECTION")
            lines.append(f"Status: {event.details.get('status', 'unknown')}")
            lines.append(f"Feedback: {event.details.get('feedback', '')}")
            
        elif event.event_type == "claim_extraction":
            lines.append(f"[{idx}] CLAIM EXTRACTION")
            lines.append(f"Extracted {event.details.get('count', 0)} claims.")

        elif event.event_type == "claim_assessment":
            lines.append(f"[{idx}] CLAIM ASSESSMENT")
            lines.append(f"Claim: {event.details.get('claim', '')}")
            lines.append(f"Supported: {event.details.get('supported', False)}")
            lines.append(f"Reason: {event.details.get('reason', '')}")
            
        elif event.event_type == "grounding_check":
            lines.append(f"[{idx}] GROUNDING CHECK")
            lines.append(f"Supported: {event.details.get('supported_claims', 0)}/{event.details.get('claim_count', 0)}")
            lines.append(f"Score: {event.details.get('grounding_score', 0.0)}")
            if event.details.get('unsupported_claims'):
                lines.append(f"Unsupported claims: {event.details.get('unsupported_claims')}")
            if event.details.get('conflicts_detected'):
                lines.append(f"Conflicts detected: {event.details.get('conflicts_detected')}")
                
        elif event.event_type == "citation_validation":
            lines.append(f"[{idx}] CITATION VALIDATION")
            if event.details.get('invalid_citations'):
                lines.append(f"Invalid citations: {event.details.get('invalid_citations')}")
            else:
                lines.append("All citations valid.")
                
        elif event.event_type == "final_answer":
            lines.append(f"[{idx}] FINAL ANSWER")
            lines.append("Completed.")
            
        elif event.event_type == "agent_error":
            lines.append(f"[{idx}] AGENT ERROR")
            lines.append(f"Error: {event.details.get('error', '')}")
            
        lines.append("")
        
    lines.append("=" * 50)
    return "\n".join(lines)
