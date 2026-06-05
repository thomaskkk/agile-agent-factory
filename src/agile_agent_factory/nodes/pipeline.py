"""LangGraph lifecycle node functions — init, PO, QA, UX, TL, refinement gate, finalize.

Node contract: accepts PipelineState, returns a partial dict that LangGraph merges
into the graph state via its reducers. Nodes never call update_state(); the
SqliteSaver checkpointer handles persistence between nodes automatically.

The dev/test/review nodes live in their own modules (dev_node.py, test_node.py,
review_node.py); shared helpers live in helpers.py. nodes/__init__.py re-exports
everything as the stable public surface that graph.py imports.
"""

from __future__ import annotations

from agile_agent_factory.config import PRODUCT_ROOT, WIP_LIMITS
from agile_agent_factory.tools.jira_client import JiraClient
from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
from agile_agent_factory.tools.logger import log
from agile_agent_factory.state import PipelineState
from agile_agent_factory.nodes.helpers import (
    _story_keys, _active_story, _safe_transition, _to_legacy_state,
    _notify_quota, _story_summary,
)


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
        gherkin, test_contracts = inject_gherkin_criteria(jira, [sk])
    except LLMQuotaExceeded as e:
        _notify_quota(jira, sk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": sk})
        gherkin, test_contracts = inject_gherkin_criteria(jira, [sk])

    criteria = gherkin.get(sk, [])
    test_contract = test_contracts.get(sk, {})

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
    ready_contracts = {
        sk: stories.get(sk, {}).get("ready_contract", {})
        for sk in story_keys
        if stories.get(sk, {}).get("ready_validated")
    }
    log(f"TL: designing architecture for {story_keys}.")

    for ek in state.get("epic_keys", []):
        _safe_transition(jira, ek, "To Tech Refinement")

    legacy = _to_legacy_state(state)
    legacy["story_keys"] = story_keys

    try:
        result = design_architecture(jira, story_keys, gherkin, ux_spec, legacy, ready_contracts=ready_contracts)
    except LLMQuotaExceeded as e:
        bk = story_keys[0] if story_keys else None
        _notify_quota(jira, bk, e)
        interrupt({"type": "quota", "provider": getattr(e, "provider", "unknown"), "blocking_key": bk})
        result = design_architecture(jira, story_keys, gherkin, ux_spec, legacy, ready_contracts=ready_contracts)

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


def refinement_gate_node(state: PipelineState) -> dict:
    """Advance a story from refinement to tech_design once its ready contract is valid.

    This is a zero-LLM deterministic gate after the parallel qa/ux fan-out.
    """
    from agile_agent_factory.agents.ready_contract import (
        build_ready_contract,
        readiness_repair_update,
        validate_ready_contract,
    )

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
        log(f"Refinement gate: {sk} not ready; keeping in refinement: {errors}")
        return {"stories": {sk: {"ready_contract": contract, **readiness_repair_update(errors)}}}

    log(f"Refinement gate: {sk} ready → tech_design.")
    return {
        "stories": {
            sk: {
                "column": "tech_design",
                "ready_contract": contract,
                "ready_validation_errors": [],
                "ready_validated": True,
            }
        }
    }


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
