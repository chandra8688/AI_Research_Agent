from dataclasses import dataclass, field
from .dataset import EvaluationCase
from agent import execute_agent
from state import AgentState

@dataclass
class EvaluationResult:
    case_name: str
    passed: bool
    skipped: bool = False
    checks: dict[str, bool] = field(default_factory=dict)

def evaluate_case(case: EvaluationCase, mock_state: AgentState = None, mock_error: Exception = None) -> EvaluationResult:
    """
    Evaluates a single case. Uses either the provided mock data (offline mode) 
    or calls the live agent if mocks are None (live mode).
    """
    error_raised = None
    state = mock_state
    
    if state is None and mock_error is None:
        try:
            _, state = execute_agent(case.query, max_iterations=2)
        except Exception as e:
            error_raised = e
    else:
        error_raised = mock_error

    checks = {}
    passed = True
    
    # 1. Error Handling Check
    if case.expected_error:
        has_error = error_raised is not None and case.expected_error in type(error_raised).__name__
        checks["error_handling"] = has_error
        passed = passed and has_error
        return EvaluationResult(case_name=case.name, passed=passed, checks=checks)
    else:
        # If we expected NO error but got one, that's a failure.
        if error_raised is not None:
            if "429" in str(error_raised) or "RESOURCE_EXHAUSTED" in str(error_raised):
                return EvaluationResult(case_name=case.name, passed=False, skipped=True, checks={"quota_exceeded": True})
            passed = False
            checks["no_unexpected_error"] = False
            return EvaluationResult(case_name=case.name, passed=passed, checks=checks)

    if not state:
        return EvaluationResult(case_name=case.name, passed=False, checks={"state_available": False})
        
    # 2. Tool Selection Accuracy
    tool_names = [tc["name"] for tc in state.tool_calls]
    tool_selection_ok = True
    for et in case.expected_tools:
        if et not in tool_names:
            tool_selection_ok = False
    for ft in case.forbidden_tools:
        if ft in tool_names:
            tool_selection_ok = False
    if case.expected_tools or case.forbidden_tools:
        checks["tool_selection"] = tool_selection_ok
        passed = passed and tool_selection_ok

    # 3. Retrieval Source Accuracy
    if case.expected_sources:
        source_ok = False
        if state.retrieved_evidence:
            combined_evidence = str(state.retrieved_evidence)
            # Just check if at least one expected source is mentioned
            for es in case.expected_sources:
                if es in combined_evidence:
                    source_ok = True
                    break
        checks["retrieval_source"] = source_ok
        passed = passed and source_ok

    # 4. Answer Completion Rate
    if case.require_non_empty_answer:
        answer_ok = state.final_answer is not None and len(state.final_answer.strip()) > 0
        checks["answer_non_empty"] = answer_ok
        passed = passed and answer_ok

    return EvaluationResult(case_name=case.name, passed=passed, checks=checks)

def print_summary(results: list[EvaluationResult]):
    total = len(results)
    passed_count = sum(1 for r in results if r.passed and not r.skipped)
    failed_count = sum(1 for r in results if not r.passed and not r.skipped)
    skipped_count = sum(1 for r in results if r.skipped)
    
    # Metrics calculation
    valid_results = [r for r in results if not r.skipped]
    valid_count = len(valid_results)
    
    pass_rate = (passed_count / valid_count * 100) if valid_count > 0 else 0
    
    # Tool selection accuracy
    tool_cases = [r for r in valid_results if "tool_selection" in r.checks]
    tool_acc = (sum(1 for r in tool_cases if r.checks["tool_selection"]) / len(tool_cases) * 100) if tool_cases else 0
    
    # Retrieval source accuracy
    retrieval_cases = [r for r in valid_results if "retrieval_source" in r.checks]
    retrieval_acc = (sum(1 for r in retrieval_cases if r.checks["retrieval_source"]) / len(retrieval_cases) * 100) if retrieval_cases else 0
    
    # Answer completion
    answer_cases = [r for r in valid_results if "answer_non_empty" in r.checks]
    answer_acc = (sum(1 for r in answer_cases if r.checks["answer_non_empty"]) / len(answer_cases) * 100) if answer_cases else 0
    
    # Error handling
    error_cases = [r for r in valid_results if "error_handling" in r.checks]
    error_acc = (sum(1 for r in error_cases if r.checks["error_handling"]) / len(error_cases) * 100) if error_cases else 0

    print("========================================")
    print("AI RESEARCH AGENT EVALUATION")
    print("========================================")
    print(f"\nTotal cases: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Pass rate: {pass_rate:.1f}%\n")
    print(f"Tool selection accuracy: {tool_acc:.1f}%")
    print(f"Retrieval source accuracy: {retrieval_acc:.1f}%")
    print(f"Answer completion rate: {answer_acc:.1f}%")
    print(f"Error handling accuracy: {error_acc:.1f}%\n")
    print("----------------------------------------")
    print("CASE RESULTS")
    print("----------------------------------------\n")
    for r in results:
        status = "SKIPPED" if r.skipped else ("PASS" if r.passed else "FAIL")
        print(f"{r.case_name:<25} {status}")
    print("\n========================================")
