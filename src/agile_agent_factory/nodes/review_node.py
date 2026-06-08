"""Code-review node — LLM audit of generated code against the DoD for ONE story."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from agile_agent_factory.config import MAX_REVIEW_RETRIES
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc, make_adf_heading
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import _active_story, _safe_transition, raise_quota_interrupt


# --------------------------------------------------------------------------- #
# Milestone 5 — Reviewer verdict classification helpers                        #
# --------------------------------------------------------------------------- #

def _is_out_of_scope_rejection(reason: str, write_scope: list[str]) -> bool:
    """Layer 1: True when the rejection only cites files outside write_scope."""
    if not write_scope or not reason:
        return False
    # Extract any file paths from the reason string
    cited = re.findall(r"(?:app|tests)/[\w/]+\.py", reason)
    if not cited:
        return False
    scope_set = set(write_scope)
    # If ALL cited files are outside the scope → out-of-scope rejection
    return all(f not in scope_set for f in cited)


def _is_vague_rejection(reason: str, story_criteria: list[str]) -> bool:
    """Layer 2: True only when the rejection has NO concrete anchor at all.

    Concrete anchors: file paths, test function names, criterion text excerpts,
    or any domain-specific technical term (test/function/class/import/assertion/criteria).
    Generic phrases with no such anchor are vague; anything with at least one anchor counts.
    """
    if not reason.strip():
        return True
    if len(reason.strip()) < 10:
        return True
    # File path or underscore_test_name pattern
    has_file_path = bool(re.search(r"(?:app|tests)/[\w/]+\.py", reason))
    has_test_name = bool(re.search(r"\btest_\w+\b", reason))
    has_criterion_text = any(
        len(first_line := c.strip().splitlines()[0]) > 6 and first_line.lower() in reason.lower()
        for c in (story_criteria or [])
    )
    # Domain terms that indicate the reviewer is pointing at something specific
    has_technical_anchor = bool(re.search(
        r"\b(test|function|class|method|import|assert|criteria|endpoint|module|"
        r"acceptance|definition|behavior|missing|failing|error|exception|return)\b",
        reason, re.IGNORECASE,
    ))
    return not (has_file_path or has_test_name or has_criterion_text or has_technical_anchor)


def _filter_rejection(
    reason: str,
    write_scope: list[str],
    story_criteria: list[str],
    story_key: str,
) -> tuple[bool, str]:
    """Apply layers 1–2 to a rejection verdict.

    Returns (should_count, filtered_reason):
      should_count=False → don't consume a retry (filtered by layer 1 or 2)
      should_count=True  → genuine rejection; consume a retry
    """
    if _is_out_of_scope_rejection(reason, write_scope):
        log(f"Review: layer-1 filter — rejection cites only out-of-scope files for {story_key}. Not counting.")
        return False, reason

    if _is_vague_rejection(reason, story_criteria):
        log(f"Review: layer-2 filter — rejection has no concrete anchor for {story_key}. Not counting.")
        return False, reason

    return True, reason


# --------------------------------------------------------------------------- #
# Milestone 3 — Deterministic pre-review gate                                 #
# --------------------------------------------------------------------------- #

def _pre_review_gate(
    story: dict,
    write_scope: list[str],
    product_root: Path,
    extra_packages: list[str] | None = None,
) -> tuple[bool, str]:
    """Deterministic structural checks run before the LLM reviewer.

    Returns (passed, reason):
      passed=True  → all checks green, proceed to LLM review
      passed=False → reason describes what failed; treat as rework trigger
    """
    from agile_agent_factory.tools.pytest_runner import run_pytest

    tc = story.get("test_contract", {}) or {}

    # Compute test_file_rel and test_file_path once, used in Checks 2 and 4
    test_file_rel = tc.get("test_file", "")
    test_file_path = product_root / test_file_rel if test_file_rel else None

    # --- Check 1: Scope completeness ---
    if write_scope:
        missing = [p for p in write_scope if not (product_root / p).exists()]
        if missing:
            return False, f"Missing required files: {missing}"

    # --- Check 2: Test function presence ---
    rc = story.get("ready_contract", {}) or {}
    expected_tests: list[str] = rc.get("expected_tests", []) or []
    if expected_tests:
        if test_file_path and test_file_path.exists():
            try:
                source = test_file_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                defined_names = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                }
                missing_fns = [fn for fn in expected_tests if fn not in defined_names]
                if missing_fns:
                    return False, f"Missing expected test functions: {missing_fns}"
            except Exception:
                # If we can't parse the test file, skip this check (don't hard-fail)
                pass

    # --- Check 3: Import contract validity (syntax check on source files) ---
    target_imports: list[str] = tc.get("target_imports", []) or []
    if target_imports:
        for imp in target_imports:
            m = re.match(r"from (app(?:\.\w+)+) import", imp)
            if not m:
                continue
            src_rel = m.group(1).replace(".", "/") + ".py"
            src_path = product_root / src_rel
            if not src_path.exists():
                continue
            try:
                source = src_path.read_text(encoding="utf-8")
                ast.parse(source)
            except SyntaxError as e:
                return False, f"Syntax error in {src_rel}: {e}"
            except Exception:
                pass

    # --- Check 4: Targeted pytest pass ---
    if test_file_rel:
        if test_file_path.exists():
            try:
                exit_code, output = run_pytest(extra_packages or [], test_targets=[str(test_file_path)])
            except Exception:
                exit_code, output = 0, ""  # can't verify; proceed to LLM reviewer
            # Exit code 5 means "no tests collected" — treat as pass
            if exit_code != 0 and exit_code != 5:
                from agile_agent_factory.nodes.dev_node import _extract_error_summary
                error_detail = _extract_error_summary(output)
                return False, f"Targeted tests failing: {error_detail}"

    return True, ""


def review_node(state: PipelineState) -> dict:
    """Reviewer agent: LLM audit of generated code against the DoD for ONE story."""
    from langgraph.types import interrupt
    from agile_agent_factory.agents.reviewer_agent import review_patch

    jira = JiraClient()
    sk, story = _active_story(state)
    review_retries = story.get("review_retries", state.get("review_retries", 0))
    log(f"Review: auditing code for {sk}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, WorkflowState.IN_CODE_REVIEW)

    story_criteria = state.get("gherkin_criteria", {}).get(sk, [])

    # Derive write scope from test_contract so the reviewer only verdicts on owned files
    tc = story.get("test_contract", {})
    write_scope: list[str] = []
    if tc:
        if tc.get("test_file"):
            write_scope.append(tc["test_file"])
        for imp in (tc.get("target_imports") or []):
            m = re.match(r"from (app(?:\.\w+)+) import", imp)
            if m:
                path_str = m.group(1).replace(".", "/") + ".py"
                if path_str not in write_scope:
                    write_scope.append(path_str)

    # Milestone 3: deterministic pre-gate — catches structural failures before LLM review
    from agile_agent_factory.config import PRODUCT_ROOT as _PRODUCT_ROOT
    from agile_agent_factory.tools.dependencies import resolve_dependencies
    from agile_agent_factory.nodes.helpers import _to_legacy_state
    deps = resolve_dependencies(_to_legacy_state(state, sk), _PRODUCT_ROOT)
    gate_passed, gate_reason = _pre_review_gate(story, write_scope, _PRODUCT_ROOT, extra_packages=deps)
    if not gate_passed:
        log(f"Review: pre-gate failed for {sk}: {gate_reason}")
        review_retries += 1
        if review_retries >= MAX_REVIEW_RETRIES:
            log(f"Max review retries ({MAX_REVIEW_RETRIES}) exhausted for {sk} (pre-gate). Triggering HITL.")
            summary = (
                f"Code review HITL: pre-gate failed after {review_retries} attempts.\n\n"
                f"Gate failure reason:\n\n{gate_reason}"
            )
            jira.add_comment_adf(sk, make_adf_doc(summary))
            jira.set_flag(sk)
            interrupt({"type": "intervention", "blocking_key": sk})
            try:
                jira.clear_flag(sk)
            except Exception:
                pass
            return {
                "review_approved": False,
                "review_retries": 0,
                "stories": {sk: {"review_retries": 0, "review_status": "pending_review", "hitl_type": "review_exhaustion"}},
            }
        cycle = review_retries
        log(f"Review pre-gate failed for {sk} (retry {cycle}/{MAX_REVIEW_RETRIES}). Keeping in code_review for rework.")
        kickback_doc = {
            "version": 1,
            "type": "doc",
            "content": [
                make_adf_heading(
                    f"Code review pre-gate failed — keeping in code_review for rework (cycle {cycle} of {MAX_REVIEW_RETRIES})"
                ),
                {"type": "paragraph", "content": [{"type": "text", "text": f"Reason: {gate_reason}"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "The downstream agent will attempt to fix the structural issues above."}]},
            ],
        }
        jira.add_comment_adf(sk, kickback_doc)
        return {
            "review_approved": False,
            "review_retries": cycle,
            "stories": {sk: {
                "column": "code_review",
                "review_status": "rework_needed",
                "review_retries": cycle,
                "review_rejection_reason": gate_reason,
            }},
        }

    try:
        result = review_patch(jira, [sk], story_criteria=story_criteria or None, story_key=sk, write_scope=write_scope or None)
    except LLMQuotaExceeded as e:
        patch = raise_quota_interrupt(jira, sk, e, state=state)
        if patch:
            return patch

    approved = result.payload.get("approved", False)
    reason = result.payload.get("reason", "")

    if not approved:
        # Milestone 5a: 3-layer filter before counting as a real rejection
        should_count, reason = _filter_rejection(reason, write_scope, story_criteria, sk)

        if not should_count:
            # Filtered — don't increment retry counter; log and proceed (treat as approved for routing)
            log(f"Review rejection filtered out for {sk} — not counting toward retry budget.")
            # Post an informational note
            try:
                jira.add_comment_adf(
                    sk,
                    make_adf_doc(
                        f"Review verdict filtered (out-of-scope or vague — see reason below). "
                        f"Story advancing without consuming retry budget.\n\nFiltered reason: {reason}"
                    ),
                )
            except Exception:
                pass
            approved = True  # treat as approved for routing purposes

    if not approved and review_retries < MAX_REVIEW_RETRIES:
        cycle = review_retries + 1
        log(f"Review rejected for {sk} (retry {cycle}/{MAX_REVIEW_RETRIES}). Keeping in code_review for rework.")
        kickback_doc = {
            "version": 1,
            "type": "doc",
            "content": [
                make_adf_heading(
                    f"Code review rejected — keeping in code_review for rework (cycle {cycle} of {MAX_REVIEW_RETRIES})"
                ),
                {"type": "paragraph", "content": [{"type": "text", "text": f"Reason: {reason}"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "The downstream agent will attempt to fix the issues above."}]},
            ],
        }
        jira.add_comment_adf(sk, kickback_doc)
        return {
            "review_approved": False,
            "review_retries": cycle,
            "stories": {sk: {
                "column": "code_review",
                "review_status": "rework_needed",
                "review_retries": cycle,
                "review_rejection_reason": reason,
            }},
        }

    if not approved and review_retries >= MAX_REVIEW_RETRIES:
        log(f"Max review retries ({MAX_REVIEW_RETRIES}) exhausted for {sk}. Triggering HITL.")
        summary = (
            f"Code review HITL: review rejected after {review_retries + 1} attempts.\n\n"
            f"Final rejection reason:\n\n{reason}"
        )
        jira.add_comment_adf(sk, make_adf_doc(summary))
        jira.set_flag(sk)
        interrupt({"type": "intervention", "blocking_key": sk})
        try:
            jira.clear_flag(sk)
        except Exception:
            pass
        # After human resume: fresh review cycle with full retry budget
        return {
            "review_approved": False,
            "review_retries": 0,
            "stories": {sk: {"review_retries": 0, "review_status": "pending_review", "hitl_type": "review_exhaustion"}},
        }

    log(f"Code review: APPROVED for {sk}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, WorkflowState.TO_QA)

    return {
        "review_approved": True,
        "stories": {sk: {"column": "done", "review_retries": review_retries, "review_status": None}},
    }
