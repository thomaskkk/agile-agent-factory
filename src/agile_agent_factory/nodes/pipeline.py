"""LangGraph lifecycle node functions — init, PO, QA, UX, TL, refinement gate, finalize.

Node contract: accepts PipelineState, returns a partial dict that LangGraph merges
into the graph state via its reducers. Nodes never call update_state(); the
SqliteSaver checkpointer handles persistence between nodes automatically.

The dev/test/review nodes live in their own modules (dev_node.py, test_node.py,
review_node.py); shared helpers live in helpers.py. nodes/__init__.py re-exports
everything as the stable public surface that graph.py imports.
"""

from __future__ import annotations

from agile_agent_factory.config import PRODUCT_ROOT, WIP_LIMITS, ASSUMPTION_RISK_THRESHOLD, MAX_REFINEMENT_RETRIES
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import (
    _story_keys, _active_story, derive_story_write_scope, _path_in_write_scope, _safe_transition, _to_legacy_state,
    _story_summary, raise_quota_interrupt,
)


def _ensure_risk_blocker(jira: JiraClient, payload: dict, risk_score: float) -> str:
    """Use an existing provisioned issue as the clarification anchor whenever possible."""
    blocking_key = (
        payload.get("blocking_issue_key")
        or next(iter(payload.get("epic_keys", [])), None)
        or next(iter(payload.get("story_keys", [])), None)
    )
    if not blocking_key:
        created = jira.create_issue(
            "HITL: High-Risk Assumption Review Needed",
            "Story",
            description_adf=make_adf_doc("High-risk assumptions require human clarification before proceeding."),
        )
        blocking_key = created["key"]

    assumptions = payload.get("assumption_ledger", []) or []
    lines = [f"- [{a.get('confidence', '?')} confidence / {a.get('impact', '?')} impact] {a.get('description', '')}" for a in assumptions]
    body = (
        f"PO paused because unresolved assumption risk is {risk_score:.2f}, above the configured threshold.\n\n"
        "Please clarify the assumptions below, then clear the Jira flag to resume.\n"
    )
    if lines:
        body += "\n" + "\n".join(lines)
    try:
        jira.add_comment_adf(blocking_key, make_adf_doc(body))
        jira.set_flag(blocking_key)
    except Exception as e:
        log(f"Failed to post PO risk blocker on {blocking_key}: {e}")
    return blocking_key


def init_node(state: PipelineState) -> dict:
    """Read business_idea.md and initialize shared pipeline state."""
    business_idea_path = PRODUCT_ROOT / "business_idea.md"
    if not business_idea_path.exists():
        raise FileNotFoundError(f"business_idea.md not found at {business_idea_path}")
    return {
        "business_idea": business_idea_path.read_text(),
        "wip_limits": state.get("wip_limits") or WIP_LIMITS,
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
        patch = raise_quota_interrupt(jira, None, e, state=state)
        if patch:
            return patch
        result = analyze_and_provision(jira, legacy)

    # Milestone 7: risk-threshold policy gate replaces binary hitl_required flag.
    hitl_required = result.payload.get("hitl_required")
    risk_score = result.payload.get("unresolved_risk_score", 0.0)
    if hitl_required:
        blocking_key = result.payload.get("blocking_issue_key")
        log(
            f"PO ambiguity detected (hitl_required={hitl_required}, risk_score={risk_score:.2f}). "
            f"Interrupting for human input. Blocking: {blocking_key}"
        )
        feedback = interrupt({"type": "refinement", "blocking_key": blocking_key})
        legacy["hitl_feedback"] = feedback or ""
        try:
            result = analyze_and_provision(jira, legacy)
        except LLMQuotaExceeded as e:
            patch = raise_quota_interrupt(jira, blocking_key, e, state=state)
            if patch:
                return patch
            result = analyze_and_provision(jira, legacy)
    elif risk_score > ASSUMPTION_RISK_THRESHOLD:
        blocking_key = _ensure_risk_blocker(jira, result.payload, risk_score)
        log(
            f"PO risk gate triggered (risk_score={risk_score:.2f} > threshold={ASSUMPTION_RISK_THRESHOLD:.2f}). "
            f"Interrupting for human input. Blocking: {blocking_key}"
        )
        interrupt({"type": "refinement", "blocking_key": blocking_key, "source": "po_risk"})

    epic_keys = result.payload.get("epic_keys", [])
    story_keys = result.payload.get("story_keys", [])
    story_to_epic = result.payload.get("story_to_epic", {})
    has_ui = result.payload.get("has_ui", False)

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
                "assumption_ledger": result.payload.get("assumption_ledger", []),
                "unresolved_risk_score": risk_score,
            }

    return {
        "epic_keys": epic_keys,
        "stories": stories,
        "has_ui": has_ui,
        "total_count": len(story_keys),
    }


def qa_node(state: PipelineState) -> dict:
    """QA agent: generate Gherkin acceptance criteria for ONE story (per-story in Phase 3+)."""
    from agile_agent_factory.agents.qa_agent import inject_gherkin_criteria

    jira = JiraClient()
    sk, story = _active_story(state)
    log(f"QA: generating Gherkin criteria for {sk}.")

    try:
        result = inject_gherkin_criteria(
            jira,
            [sk],
            story_feedback={sk: story.get("hitl_feedback", "")} if story.get("hitl_feedback") else None,
        )
    except LLMQuotaExceeded as e:
        patch = raise_quota_interrupt(jira, sk, e, state=state)
        if patch:
            return patch
        result = inject_gherkin_criteria(jira, [sk])

    criteria = result.payload["gherkin_criteria"].get(sk, [])
    test_contract = result.payload["test_contracts"].get(sk, {})

    # Just set the flag — the dispatcher/refinement_gate handles column advancement
    # so parallel QA+UX dispatch doesn't race on the column field.
    return {
        "gherkin_criteria": {sk: criteria},
        "stories": {
            sk: {
                "gherkin_criteria": criteria,
                "test_contract": test_contract,
                "refinement_qa_done": True,
            }
        },
    }


def ux_node(state: PipelineState) -> dict:
    """UX agent: design screens/flows for ONE story (only when has_ui is True)."""
    from agile_agent_factory.agents.ux_agent import design_user_experience

    jira = JiraClient()
    sk, story = _active_story(state)
    gherkin = {sk: story.get("gherkin_criteria", state.get("gherkin_criteria", {}).get(sk, []))}
    legacy = _to_legacy_state(state, sk)
    log(f"UX: designing experience for {sk}.")

    try:
        spec = design_user_experience(
            jira,
            [sk],
            gherkin,
            legacy,
            story_feedback={sk: story.get("hitl_feedback", "")} if story.get("hitl_feedback") else None,
        ).payload["ux_spec"]
    except LLMQuotaExceeded as e:
        patch = raise_quota_interrupt(jira, sk, e, state=state)
        if patch:
            return patch
        spec = design_user_experience(jira, [sk], gherkin, legacy).payload["ux_spec"]

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
    ready_contracts = {
        sk: stories.get(sk, {}).get("ready_contract", {})
        for sk in story_keys
        if stories.get(sk, {}).get("ready_validated")
    }
    log(f"TL: designing architecture for {story_keys}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, WorkflowState.TECH_REFINEMENT)

    legacy = _to_legacy_state(state)
    legacy["story_keys"] = story_keys

    try:
        result = design_architecture(jira, story_keys, gherkin, ux_spec, legacy, ready_contracts=ready_contracts)
    except LLMQuotaExceeded as e:
        bk = story_keys[0] if story_keys else None
        patch = raise_quota_interrupt(jira, bk, e, state=state)
        if patch:
            return patch
        result = design_architecture(jira, story_keys, gherkin, ux_spec, legacy, ready_contracts=ready_contracts)

    payload = result.payload
    arch = payload.get("architecture", {})
    subtasks = payload.get("subtasks", {})
    deps = payload.get("dependencies", [])

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


def refinement_gate_node(state: PipelineState) -> dict:
    """Advance a story from refinement to tech_design once its ready contract is valid.

    This is a zero-LLM deterministic gate after the parallel qa/ux fan-out.
    Caps retries at MAX_REFINEMENT_RETRIES; exhaustion triggers HITL.
    """
    from langgraph.types import interrupt
    from agile_agent_factory.agents.ready_contract import (
        build_ready_contract,
        readiness_repair_update,
        validate_ready_contract,
    )
    from agile_agent_factory.tools.jira_client import make_adf_doc

    jira = JiraClient()
    sk, story = _active_story(state)
    criteria = story.get("gherkin_criteria", state.get("gherkin_criteria", {}).get(sk, []))
    ux_spec = story.get("ux_spec") or state.get("ux_spec", {})
    contract = build_ready_contract(
        story_key=sk,
        story=story,
        summary=_story_summary(jira, sk),
        business_idea=state.get("business_idea", ""),
        acceptance_criteria=criteria,
        ux_spec=ux_spec,
        test_contract=story.get("test_contract"),
    )
    errors = validate_ready_contract(contract)

    if errors:
        refinement_retries = story.get("refinement_retries", 0) + 1
        log(f"Refinement gate: {sk} not ready (attempt {refinement_retries}/{MAX_REFINEMENT_RETRIES}): {errors}")

        if refinement_retries >= MAX_REFINEMENT_RETRIES:
            log(f"Refinement retries exhausted for {sk}. Triggering HITL.")
            try:
                jira.add_comment_adf(
                    sk,
                    make_adf_doc(
                        f"Refinement gate failed {refinement_retries} times — human review needed.\n\n"
                        f"Errors: {errors}"
                    ),
                )
                jira.set_flag(sk)
            except Exception:
                pass
            interrupt({
                "type": "intervention",
                "blocking_key": sk,
                "source": "refinement_gate",
                "reason": f"Refinement gate exhausted after {refinement_retries} attempts",
                "errors": errors,
            })
            return {
                "stories": {
                    sk: {
                        "ready_contract": contract,
                        "refinement_retries": 0,
                        "hitl_type": "intervention",
                        **readiness_repair_update(errors),
                    }
                }
            }

        return {
            "stories": {
                sk: {
                    "ready_contract": contract,
                    "refinement_retries": refinement_retries,
                    **readiness_repair_update(errors),
                }
            }
        }

    log(f"Refinement gate: {sk} ready → tech_design.")
    return {
        "stories": {
            sk: {
                "column": "tech_design",
                "ready_contract": contract,
                "ready_validation_errors": [],
                "ready_validated": True,
                "refinement_retries": 0,
                "hitl_feedback": "",
            }
        }
    }


def finalize_node(state: PipelineState) -> dict:
    """Run README generation and SRE deployment once all stories are done."""
    from agile_agent_factory.agents.readme_agent import generate_readme
    from agile_agent_factory.agents.sre_agent import emulate_deployment
    from agile_agent_factory.tools.jira_client import make_adf_doc

    jira = JiraClient()
    story_keys = _story_keys(state)
    legacy = _to_legacy_state(state)
    # --- Blocker resolution pass ---
    # Collect stories whose test_node quarantined cross-story regressions.
    all_stories = state.get("stories", {})
    blocker_stories = {
        sk: story
        for sk, story in all_stories.items()
        if story.get("regression_blockers")
    }

    # Build a map: file path → owning story key (the story whose write_scope covers that file).
    def _owner_for_file(path: str) -> str | None:
        for candidate_sk, candidate_story in all_stories.items():
            if _path_in_write_scope(path, derive_story_write_scope(candidate_sk, candidate_story)):
                return candidate_sk
        return None

    requeue_updates: dict[str, dict] = {}
    stories_update: dict[str, dict] = {}
    unresolved_count = 0
    fallback_warned_count = 0
    for quarantine_sk, story in blocker_stories.items():
        unresolved_for_story: list[str] = []
        for blocker_file in story.get("regression_blockers", []):
            owning_sk = _owner_for_file(blocker_file)
            if owning_sk is None:
                # No owner found — truly unresolved, no Jira comment to post.
                unresolved_count += 1
                unresolved_for_story.append(blocker_file)
                log(
                    f"Finalize: unresolved regression blocker {blocker_file!r} "
                    f"(quarantined by {quarantine_sk}); no owning story found."
                )
                continue
            owning_story = all_stories.get(owning_sk, {})
            owning_column = owning_story.get("column", "")
            if owning_column in {"done", "code_review"}:
                owner_update = requeue_updates.setdefault(owning_sk, {
                    "column": "testing",
                    "review_status": None,
                    "review_retries": 0,
                    "review_rejection_reason": "",
                    "hitl_type": None,
                    "incoming_regression_files": list(owning_story.get("incoming_regression_files", [])),
                    "incoming_regression_output": owning_story.get("incoming_regression_output", ""),
                })
                incoming_files = owner_update.get("incoming_regression_files", [])
                if blocker_file not in incoming_files:
                    owner_update["incoming_regression_files"] = incoming_files + [blocker_file]
                note = (
                    f"Finalize detected a cross-story regression from {quarantine_sk} "
                    f"touching {blocker_file}. Auto-requeued to testing."
                )
                existing_output = owner_update.get("incoming_regression_output", "")
                if note not in existing_output:
                    owner_update["incoming_regression_output"] = (
                        f"{existing_output}\n\n---\n\n{note}".strip() if existing_output else note
                    )
                log(
                    f"Finalize: requeueing owner {owning_sk} to testing for blocker {blocker_file!r} "
                    f"(quarantined by {quarantine_sk})."
                )
                continue
            if owning_column != "done":
                # The owning story hasn't finished yet; it will pick up the regression
                # naturally when it reaches testing. Skip silently.
                unresolved_for_story.append(blocker_file)
                log(
                    f"Finalize: blocker {blocker_file!r} owned by {owning_sk} "
                    f"(column={owning_column!r}); will resolve naturally — skipping."
                )
                continue
        if story.get("regression_blockers"):
            stories_update[quarantine_sk] = {"regression_blockers": unresolved_for_story}

    if requeue_updates:
        for owning_sk, owner_update in requeue_updates.items():
            note = (
                "Finalize automatically reopened this story for testing because a "
                "cross-story regression was detected in: "
                f"{', '.join(owner_update.get('incoming_regression_files', []))}."
            )
            try:
                jira.add_comment_adf(owning_sk, make_adf_doc(note))
            except Exception as exc:  # noqa: BLE001
                log(f"Finalize: failed to post auto-requeue note on {owning_sk}: {exc}")
        log(
            f"Finalize: deferred completion and requeued {len(requeue_updates)} "
            "owner story/stories back to testing."
        )
        stories_update.update(requeue_updates)
        return {"stories": stories_update}

    log("Finalize: generating README and running deployment.")

    generate_readme(legacy)
    emulate_deployment(jira, story_keys, legacy)

    done_updates = {sk: {"column": "done"} for sk in story_keys}
    for sk, patch in stories_update.items():
        done_updates[sk] = {**done_updates.get(sk, {}), **patch}

    if unresolved_count > 0:
        for quarantine_sk, story in blocker_stories.items():
            unresolved_files = done_updates.get(quarantine_sk, {}).get("regression_blockers", [])
            if not unresolved_files:
                continue
            fallback_warned_count += 1
            warning_text = (
                "Finalize could not identify an owning story for cross-story regression file(s): "
                f"{', '.join(unresolved_files)}. Manual review required."
            )
            try:
                jira.add_comment_adf(quarantine_sk, make_adf_doc(warning_text))
            except Exception as exc:  # noqa: BLE001
                log(f"Finalize: failed to post fallback warning on {quarantine_sk}: {exc}")

    summary = f"Pipeline finalized: {len(story_keys)} story/stories done."
    parts = []
    if fallback_warned_count > 0:
        parts.append(f"{fallback_warned_count} unresolved regression warning(s) posted to Jira")
    if unresolved_count > 0:
        parts.append(f"{unresolved_count} cross-story regression(s) with no identifiable owner")
    if parts:
        summary += "\n\nRegression blockers: " + "; ".join(parts) + "."
    log(summary)

    return {
        "stories": done_updates,
        "done_count": len(story_keys),
    }
