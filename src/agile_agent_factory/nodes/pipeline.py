"""LangGraph node functions — each wraps an existing agent for use in the StateGraph.

Node contract: accepts PipelineState, returns a partial dict that LangGraph merges
into the graph state via its reducers. Nodes never call update_state(); the
SqliteSaver checkpointer handles persistence between nodes automatically.
"""

from __future__ import annotations

import json

from agile_agent_factory.config import (
    MAX_CORRECTION_FAILURES, MAX_RETRIES_DEV, MAX_REVIEW_RETRIES,
    PRODUCT_ROOT, BLUEPRINT_PATH,
    DEV_MODEL, TEST_MODEL,
)
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_bullet_list, make_adf_doc, make_adf_heading
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded, call_llm_json
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.path_utils import normalize_generated_path
from agile_agent_factory.tools.dependencies import resolve_dependencies
from agile_agent_factory.tools.pytest_runner import run_pytest
from agile_agent_factory.state import PipelineState

DEFAULT_WIP_LIMITS = {
    "refinement": 3,
    "tech_design": 2,
    "development": 2,
    "testing": 2,
    "code_review": 1,
}

_FILE_FALLBACK = [
    {"path": "app/__init__.py", "content": ""},
    {"path": "tests/__init__.py", "content": ""},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _story_keys(state: PipelineState) -> list[str]:
    return list(state.get("stories", {}).keys())


def _active_story(state: PipelineState) -> tuple[str, dict]:
    """Return (story_key, story_dict) for the active story in this node invocation."""
    sk = state.get("active_story_key") or _story_keys(state)[0]
    story = state.get("stories", {}).get(sk, {})
    return sk, story


def _safe_transition(jira: JiraClient, key: str, target: str) -> None:
    try:
        jira.transition_issue(key, target)
    except ValueError as e:
        log(str(e))


def _to_legacy_state(state: PipelineState, story_key: str | None = None) -> dict:
    """Build a legacy flat-state dict compatible with existing agent interfaces."""
    stories = state.get("stories", {})
    story: dict = {}
    if story_key:
        story = stories.get(story_key, {})

    return {
        "status": "READY",
        "current_phase": None,
        "blocking_issue_key": None,
        "hitl_feedback": story.get("hitl_feedback", ""),
        "epic_keys": state.get("epic_keys", []),
        "story_keys": _story_keys(state),
        "subtasks": story.get("subtasks") or state.get("subtasks", {}),
        "review_retries": story.get("review_retries", state.get("review_retries", 0)),
        "has_ui": story.get("has_ui", state.get("has_ui", False)),
        "dependencies": story.get("dependencies") or state.get("dependencies", []),
        "gherkin_criteria": {story_key: story.get("gherkin_criteria", [])} if story_key else state.get("gherkin_criteria", {}),
        "ux_spec": story.get("ux_spec") or state.get("ux_spec", {}),
        "architecture": story.get("architecture") or state.get("architecture", {}),
    }


def _notify_quota(jira: JiraClient, issue_key: str | None, exc: LLMQuotaExceeded) -> None:
    from agile_agent_factory.config import JIRA_HUMAN_ACCOUNT_ID
    from agile_agent_factory.tools.jira_client import make_adf_mention_doc
    provider = getattr(exc, "provider", "unknown")
    if issue_key:
        try:
            jira.set_flag(issue_key)
            jira.add_comment_adf(
                issue_key,
                make_adf_mention_doc(
                    JIRA_HUMAN_ACCOUNT_ID,
                    f"LLM quota exceeded (provider: {provider}). "
                    "Resolve the quota limit and clear this flag to resume the pipeline.",
                ),
            )
        except Exception as notify_err:
            log(f"Failed to post quota notification on {issue_key}: {notify_err}")
    log(f"Quota exceeded ({provider}). Pipeline pausing for human intervention.")


def _write_generated_files(files: list) -> list[str]:
    written: list[str] = []
    for f in files:
        try:
            target = normalize_generated_path(f["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.get("content", ""))
            log(f"Wrote: {target}")
            written.append(f["path"])
        except (ValueError, KeyError) as e:
            log(f"Skipped invalid path {f.get('path', '?')}: {e}")
    return written


def _generate_code_with_llm(blueprint: str, review_feedback: str = "", model: str | None = None) -> None:
    system = (
        "You are a senior Python developer implementing a feature from a technical blueprint. "
        "Return a JSON list of files to write. Use only paths starting with app/ or tests/. "
        "Never use absolute paths or nested app/app/ paths."
    )
    feedback_section = (
        f"\n\nPrevious code review rejected this implementation. You MUST fix all of the following:\n{review_feedback}"
        if review_feedback else ""
    )
    prompt = f"""Implement the following blueprint. Return JSON only:
[
  {{"path": "app/module.py", "content": "# full file content here"}},
  ...
]

Blueprint:
{blueprint}{feedback_section}
"""
    files = call_llm_json(prompt, system=system, fallback=_FILE_FALLBACK, model=model)
    _write_generated_files(files)


def _correct_code(blueprint: str, traceback: str, model: str | None = None) -> list[str]:
    """Ask the LLM to fix failing tests. Returns list of written file paths, or []."""
    generated: dict[str, str] = {}
    for target_dir in ("app", "tests"):
        d = PRODUCT_ROOT / target_dir
        if d.exists():
            for f in sorted(d.rglob("*.py")):
                rel = str(f.relative_to(PRODUCT_ROOT))
                generated[rel] = f.read_text()

    files_block = "\n\n".join(
        f"### {path}\n```python\n{content[:6000]}\n```"
        for path, content in list(generated.items())[:20]
    )

    system = (
        "You are a Python developer fixing a failing test suite. "
        "Your ENTIRE response must be ONLY a valid JSON array — "
        "no other text, no explanation, no markdown fencing outside the array. "
        'Each element: {"path": "app/…", "content": "full file content"}. '
        "Return ONLY the files you must change to fix the error — do NOT return files "
        "that are already correct. Return the full content of each file you do change. "
        "Use only paths starting with app/ or tests/. "
        "Do NOT create requirements.txt, setup.py, setup.cfg, pyproject.toml or other "
        "config/dependency files — dependencies are managed separately."
    )
    prompt = f"""[
  {{"path": "app/file_to_fix.py", "content": "FULL CORRECTED CONTENT HERE"}}
]

Replace the template above with ONLY the files that must change to fix the failing
tests below. Do not include unchanged files. Do not add any text before or after the JSON array.

Traceback:
{traceback}

Current source files:
{files_block}
"""
    try:
        files = call_llm_json(prompt, system=system, model=model)
    except json.JSONDecodeError:
        log("LLM correction produced unparseable response — no files written.")
        return []
    if isinstance(files, dict):
        files = [files]
    if not isinstance(files, list) or not all(isinstance(f, dict) for f in files):
        log(f"LLM correction returned unexpected structure ({type(files).__name__}) — no files written.")
        return []
    if not files:
        log("LLM correction returned empty file list — no files written.")
        return []
    written = _write_generated_files(files)
    log(f"Correction applied: {len(written)} file(s) updated.")
    return written


def _extract_error_summary(output: str) -> str:
    prefixes = ("FAILED ", "ERROR ", "AssertionError", "assert ", "E   ", "E ")
    lines = []
    for line in output.splitlines():
        if any(line.startswith(p) for p in prefixes):
            lines.append(line.rstrip())
            if len(lines) >= 3:
                break
    summary = "\n".join(lines)
    return summary[:400] if summary else output[-400:].strip()


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def init_node(state: PipelineState) -> dict:
    """Read business_idea.md and initialize shared pipeline state."""
    business_idea_path = PRODUCT_ROOT / "business_idea.md"
    if not business_idea_path.exists():
        raise FileNotFoundError(f"business_idea.md not found at {business_idea_path}")
    return {
        "business_idea": business_idea_path.read_text(),
        "wip_limits": state.get("wip_limits") or DEFAULT_WIP_LIMITS,
        "review_retries": 0,
        "review_approved": False,
        "done_count": 0,
    }


def po_node(state: PipelineState) -> dict:
    """PO agent: analyze business idea, create Jira epics + stories.

    Interrupts if the LLM flags ambiguity in requirements.
    On resume, re-runs with human feedback injected into the prompt.
    """
    from langgraph.types import interrupt
    from agile_agent_factory.agents.po_agent import analyze_and_provision

    jira = JiraClient()

    if state.get("stories"):
        log("Stories already created — skipping PO node (idempotency guard).")
        return {}

    legacy = {"story_keys": [], "epic_keys": [], "hitl_feedback": ""}

    try:
        result = analyze_and_provision(jira, legacy)
    except LLMQuotaExceeded as e:
        _notify_quota(jira, None, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": None})
        result = analyze_and_provision(jira, legacy)

    if result.get("hitl_required"):
        blocking_key = result.get("blocking_issue_key")
        log(f"PO ambiguity detected. Interrupting for human input. Blocking: {blocking_key}")
        feedback = interrupt({"type": "refinement", "blocking_key": blocking_key})
        legacy["hitl_feedback"] = feedback or ""
        try:
            result = analyze_and_provision(jira, legacy)
        except LLMQuotaExceeded as e:
            _notify_quota(jira, blocking_key, e)
            interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": blocking_key})
            result = analyze_and_provision(jira, legacy)

    epic_keys = result.get("epic_keys", [])
    story_keys = result.get("story_keys", [])
    story_to_epic = result.get("story_to_epic", {})
    has_ui = result.get("has_ui", False)

    stories = {}
    for sk in story_keys:
        ek = story_to_epic.get(sk, epic_keys[0] if epic_keys else "")
        stories[sk] = {
            "story_key": sk,
            "epic_key": ek,
            "column": "refinement",
            "has_ui": has_ui,
            "refinement_qa_done": False,
            "refinement_ux_done": not has_ui,  # skip UX sub-phase when no UI
        }

    return {
        "epic_keys": epic_keys,
        "stories": stories,
        "has_ui": has_ui,
        "total_count": len(story_keys),
    }


def qa_node(state: PipelineState) -> dict:
    """QA agent: generate Gherkin acceptance criteria for ONE story (per-story in Phase 3+)."""
    from langgraph.types import interrupt
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"QA: generating Gherkin criteria for {sk}.")

    try:
        gherkin = inject_gherkin_criteria(jira, [sk])
    except LLMQuotaExceeded as e:
        _notify_quota(jira, sk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": sk})
        gherkin = inject_gherkin_criteria(jira, [sk])

    criteria = gherkin.get(sk, [])

    # Just set the flag — the dispatcher/refinement_gate handles column advancement
    # so parallel QA+UX dispatch doesn't race on the column field.
    return {
        "gherkin_criteria": {sk: criteria},
        "stories": {
            sk: {
                "gherkin_criteria": criteria,
                "refinement_qa_done": True,
            }
        },
    }


def ux_node(state: PipelineState) -> dict:
    """UX agent: design screens/flows for ONE story (only when has_ui is True)."""
    from langgraph.types import interrupt
    from agile_agent_factory.agents.ux_agent import design_user_experience

    jira = JiraClient()
    sk, story = _active_story(state)
    gherkin = {sk: story.get("gherkin_criteria", state.get("gherkin_criteria", {}).get(sk, []))}
    legacy = _to_legacy_state(state, sk)
    log(f"UX: designing experience for {sk}.")

    try:
        spec = design_user_experience(jira, [sk], gherkin, legacy)
    except LLMQuotaExceeded as e:
        _notify_quota(jira, sk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": sk})
        spec = design_user_experience(jira, [sk], gherkin, legacy)

    # Just set the flag — dispatcher/refinement_gate advances the column.
    return {
        "ux_spec": spec,
        "stories": {
            sk: {
                "ux_spec": spec,
                "refinement_ux_done": True,
            }
        },
    }


def tl_node(state: PipelineState) -> dict:
    """TL agent: design architecture for all stories in tech_design (batch).

    TL stays batch because architecture is a single unified blueprint that
    spans all stories. Per-story TL dispatch is a Phase 4+ concern.
    """
    from langgraph.types import interrupt
    from agile_agent_factory.agents.tl_agent import design_architecture

    jira = JiraClient()
    stories = state.get("stories", {})

    # Collect all stories currently in tech_design column
    tech_stories = [sk for sk, s in stories.items() if s.get("column") == "tech_design"]
    if not tech_stories:
        log("TL: no stories in tech_design — skipping.")
        return {}

    story_keys = tech_stories
    gherkin = state.get("gherkin_criteria", {})
    ux_spec = state.get("ux_spec", {})
    log(f"TL: designing architecture for {story_keys}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, "To Tech Refinement")

    legacy = _to_legacy_state(state)
    legacy["story_keys"] = story_keys

    try:
        result = design_architecture(jira, story_keys, gherkin, ux_spec, legacy)
    except LLMQuotaExceeded as e:
        bk = story_keys[0] if story_keys else None
        _notify_quota(jira, bk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": bk})
        result = design_architecture(jira, story_keys, gherkin, ux_spec, legacy)

    arch = result.get("architecture", {})
    subtasks = result.get("subtasks", {})
    deps = result.get("dependencies", [])

    # Advance all processed stories to development
    stories_update = {
        sk: {"column": "development", "architecture": arch, "subtasks": subtasks, "dependencies": deps}
        for sk in story_keys
    }

    return {
        "architecture": arch,
        "subtasks": subtasks,
        "dependencies": deps,
        "stories": stories_update,
    }


def dev_node(state: PipelineState) -> dict:
    """Developer agent: generate code for ONE story."""
    from langgraph.types import interrupt

    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"Dev: generating code for {sk}.")

    is_rework = story.get("column") == "code_review"

    if not is_rework:
        _safe_transition(jira, sk, "Development")
        for ek in state.get("epic_keys", []):
            _safe_transition(jira, ek, "Development")
        for subtask_key in (story.get("subtasks") or state.get("subtasks", {})).values():
            _safe_transition(jira, subtask_key, "Development")

    blueprint = BLUEPRINT_PATH.read_text() if BLUEPRINT_PATH.exists() else ""
    review_feedback = story.get("review_rejection_reason", "")

    try:
        from agile_agent_factory.tools.aider_client import is_available, run_task
        if is_available():
            log("Using aider for code generation.")
            run_task(
                "Implement the product described in the blueprint. "
                "Write code to app/ and tests to tests/.",
                blueprint,
                review_feedback=review_feedback,
            )
        else:
            log("Aider unavailable — using LLM-direct code generation.")
            _generate_code_with_llm(blueprint, review_feedback=review_feedback, model=DEV_MODEL or None)
    except LLMQuotaExceeded as e:
        _notify_quota(jira, sk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": sk})
        _generate_code_with_llm(blueprint, review_feedback=review_feedback, model=DEV_MODEL or None)

    if is_rework:
        return {
            "stories": {sk: {
                "review_status": "pending_review",
                "review_rejection_reason": "",
            }},
        }
    return {
        "stories": {sk: {"column": "testing", "retries": 0, "correction_failures": 0}},
    }


def test_node(state: PipelineState) -> dict:
    """Run pytest with a correction loop for ONE story.

    On HITL resume, re-runs pytest from scratch (retries reset to 0).
    """
    from langgraph.types import interrupt

    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"Test: running pytest for {sk}.")

    blueprint = BLUEPRINT_PATH.read_text() if BLUEPRINT_PATH.exists() else ""
    deps = resolve_dependencies(_to_legacy_state(state, sk), PRODUCT_ROOT)

    retries = 0
    correction_failures = 0
    last_output: str | None = None

    while True:
        if last_output is None:
            exit_code, output = run_pytest(deps)
        else:
            exit_code, output = 1, last_output

        if exit_code == 0:
            log(f"All tests passed for {sk}. Moving to code review.")
            if retries > 0:
                jira.add_comment_adf(
                    sk,
                    make_adf_doc(
                        f"Tests passing after {retries} correction attempt(s). Proceeding to code review."
                    ),
                )
            _safe_transition(jira, sk, "To Code Review")
            for ek in state.get("epic_keys", []):
                _safe_transition(jira, ek, "To Code Review")
            return {"stories": {sk: {"column": "code_review"}}}

        if exit_code in (4, 5) and last_output is None:
            log(f"pytest exit code {exit_code}: collection error.")
            output += (
                f"\n\nNote: pytest exit code {exit_code} — tests could not be collected. "
                "This is an import or syntax error in the test or app modules shown above. "
                "Fix the import/definition mismatch (or generate the missing module/test) so collection succeeds."
            )

        attempt = retries + 1
        log(f"pytest failed (attempt {attempt}/{MAX_RETRIES_DEV + 1}) for {sk}.")

        if retries >= MAX_RETRIES_DEV:
            log(f"Max retries exceeded for {sk}. Triggering HITL.")
            summary = (
                f"Downstream HITL: pytest failed after {attempt} attempts.\n\n"
                f"Final traceback:\n\n{output[-3000:]}"
            )
            jira.add_comment_adf(sk, make_adf_doc(summary))
            jira.set_flag(sk)
            interrupt({"type": "intervention", "blocking_key": sk})
            try:
                jira.clear_flag(sk)
            except Exception:
                pass
            retries = 0
            correction_failures = 0
            last_output = None
            continue

        log("Requesting LLM-driven correction.")
        corrected = _correct_code(blueprint, output, model=TEST_MODEL or None)

        if not corrected:
            if correction_failures >= MAX_CORRECTION_FAILURES:
                log(f"Correction failed {correction_failures + 1}x for {sk} — escalating to HITL.")
                summary = (
                    f"HITL: correction loop failed {correction_failures + 1} times without producing valid files.\n\n"
                    f"Last traceback:\n\n{output[-3000:]}"
                )
                jira.add_comment_adf(sk, make_adf_doc(summary))
                jira.set_flag(sk)
                interrupt({"type": "intervention", "blocking_key": sk})
                try:
                    jira.clear_flag(sk)
                except Exception:
                    pass
                retries = 0
                correction_failures = 0
                last_output = None
                continue
            correction_failures += 1
            log(f"No correction produced (failure {correction_failures}/{MAX_CORRECTION_FAILURES + 1}) — retrying.")
            last_output = output
            continue

        if sk:
            error_summary = _extract_error_summary(output)
            comment_nodes = [
                make_adf_heading(f"Correction applied — attempt {attempt} of {MAX_RETRIES_DEV + 1}"),
                {"type": "paragraph", "content": [{"type": "text", "text": f"Error addressed:\n{error_summary}"}]},
                make_adf_heading("Files updated", level=4),
                make_adf_bullet_list(corrected),
            ]
            jira.add_comment_adf(sk, {"version": 1, "type": "doc", "content": comment_nodes})

        retries += 1
        correction_failures = 0
        last_output = None


def review_node(state: PipelineState) -> dict:
    """Reviewer agent: LLM audit of generated code against the DoD for ONE story."""
    from langgraph.types import interrupt
    from agile_agent_factory.agents.reviewer_agent import review_patch

    jira = JiraClient()
    sk, story = _active_story(state)
    review_retries = story.get("review_retries", state.get("review_retries", 0))
    log(f"Review: auditing code for {sk}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, "In Code Review")

    try:
        result = review_patch(jira, [sk])
    except LLMQuotaExceeded as e:
        _notify_quota(jira, sk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": sk})
        result = review_patch(jira, [sk])

    approved = result.get("approved", False)
    reason = result.get("reason", "")

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
            "stories": {sk: {"review_retries": 0, "review_status": "pending_review"}},
        }

    log(f"Code review: APPROVED for {sk}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, "To QA")

    return {
        "review_approved": True,
        "stories": {sk: {"column": "done", "review_retries": review_retries, "review_status": None}},
    }


def refinement_gate_node(state: PipelineState) -> dict:
    """Advance a story from refinement to tech_design once QA and UX are both done.

    This is a zero-LLM pass-through that exists solely to set column="tech_design"
    after the parallel qa/ux fan-out completes, avoiding a column-field race condition
    where both qa_node and ux_node would each try to advance the column simultaneously.
    """
    sk, _ = _active_story(state)
    log(f"Refinement gate: {sk} → tech_design.")
    return {"stories": {sk: {"column": "tech_design"}}}


def finalize_node(state: PipelineState) -> dict:
    """Run README generation and SRE deployment once all stories are done."""
    from agile_agent_factory.agents.readme_agent import generate_readme
    from agile_agent_factory.agents.sre_agent import emulate_deployment

    jira = JiraClient()
    story_keys = _story_keys(state)
    legacy = _to_legacy_state(state)
    log("Finalize: generating README and running deployment.")

    generate_readme(legacy)
    emulate_deployment(jira, story_keys, legacy)

    stories_update = {sk: {"column": "done"} for sk in story_keys}
    return {
        "stories": stories_update,
        "done_count": len(story_keys),
    }
