from agile_agent_factory.agents.contract import AgentResult
from agile_agent_factory.config import JIRA_HUMAN_ACCOUNT_ID, PRODUCT_ROOT, BP_BUSINESS_INTENT
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc, make_adf_mention_doc, make_adf_heading, make_adf_bullet_list
from agile_agent_factory.tools.jira_facade import JiraFacade
from agile_agent_factory.tools.llm_adapters.po import analyze_business
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState

BUSINESS_IDEA_PATH = PRODUCT_ROOT / "business_idea.md"

_HITL_FALLBACK = {
    "has_ambiguity": False,
    "hitl_required": False,
    "ambiguity_description": "",
    "assumptions": [],
    "has_ui": False,
    "epics": [],
}

# Legacy string-based confidence/impact levels (for backward compatibility with older LLM outputs)
_LEGACY_CONF = {"low": 0.3, "medium": 0.5, "high": 0.8}


def _to_float(val, default: float = 0.5) -> float:
    """
    Coerce a value to float, supporting both numeric and legacy string formats.

    - Numeric (int/float): returned as-is
    - String: mapped via _LEGACY_CONF; unrecognised strings default to 0.5
    - Other types: default to 0.5
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return _LEGACY_CONF.get(val.lower(), default)
    return default


def read_business_idea() -> str:
    log(f"Reading business idea from {BUSINESS_IDEA_PATH}.")
    if not BUSINESS_IDEA_PATH.exists():
        raise FileNotFoundError(f"business_idea.md not found at {BUSINESS_IDEA_PATH}")
    return BUSINESS_IDEA_PATH.read_text()


def analyze_and_provision(jira: JiraClient, state: dict) -> AgentResult:
    if state.get("story_keys"):
        log("Story keys already in state — skipping Jira provisioning (idempotency guard).")
        return AgentResult(payload=state)

    idea = read_business_idea()

    hitl_feedback = state.get("hitl_feedback", "")
    result = analyze_business(idea, hitl_feedback, _HITL_FALLBACK)

    # Milestone 7: explicit hitl_required flag takes priority; fall back to has_ambiguity for compat
    must_escalate = result.get("hitl_required") or result.get("has_ambiguity")
    if must_escalate:
        return AgentResult(payload=_handle_upstream_hitl(jira, result.get("ambiguity_description", "Unknown ambiguity"), state))

    # Record non-critical assumptions in the ledger; post as Jira notification
    assumptions = [a for a in (result.get("assumptions") or []) if isinstance(a, dict)]

    epic_keys: list[str] = []
    story_keys: list[str] = []
    story_to_epic: dict[str, str] = {}

    for epic_data in result.get("epics", []):
        epic = jira.create_issue(
            epic_data["title"],
            "Epic",
            description_adf=make_adf_doc(epic_data.get("description", "")),
        )
        epic_key = epic["key"]
        epic_keys.append(epic_key)
        log(f"Created Epic: {epic_key} — {epic_data['title']}")
        try:
            jira.transition_to(epic_key, WorkflowState.BUSINESS_REFINEMENT)
        except ValueError as e:
            log(str(e))

        for story_data in epic_data.get("stories", []):
            dod_items = story_data.get("definition_of_done", [])
            desc_nodes = [
                {"type": "paragraph", "content": [{"type": "text", "text": story_data.get("description", "")}]},
            ]
            if dod_items:
                desc_nodes.append(make_adf_heading("Definition of Done"))
                desc_nodes.append(make_adf_bullet_list(dod_items))
            story = jira.create_issue(
                story_data["title"],
                "Story",
                description_adf={"version": 1, "type": "doc", "content": desc_nodes},
                parent_key=epic_key,
            )
            story_keys.append(story["key"])
            story_to_epic[story["key"]] = epic_key
            log(f"Created Story: {story['key']} — {story_data['title']}")
            try:
                jira.transition_to(story["key"], WorkflowState.BUSINESS_REFINEMENT)
            except ValueError as e:
                log(str(e))

    _write_business_intent(idea, result.get("has_ui", False), epic_keys, story_keys)

    # Post assumption ledger as Jira notification when assumptions were made
    if assumptions and epic_keys:
        _post_assumption_ledger(jira, epic_keys[0], assumptions)

    # Compute unresolved risk score: impact-weighted mean of (1 - confidence).
    # High-impact + low-confidence assumptions drive the score up; low-impact ambiguity
    # does not inflate the score needlessly.
    # Uses _to_float to handle both numeric and legacy string-based confidence/impact.
    risk_score = (
        sum(_to_float(a.get("impact")) * (1 - _to_float(a.get("confidence"))) for a in assumptions)
        / max(len(assumptions), 1)
        if assumptions else 0.0
    )

    return AgentResult(payload={
        **state,
        "current_phase": "upstream_po_done",
        "epic_keys": epic_keys,
        "story_keys": story_keys,
        "has_ui": result.get("has_ui", False),
        "story_to_epic": story_to_epic,
        "assumption_ledger": assumptions,
        "unresolved_risk_score": risk_score,
    })


def _write_business_intent(idea: str, has_ui: bool, epic_keys: list, story_keys: list) -> None:
    BP_BUSINESS_INTENT.parent.mkdir(parents=True, exist_ok=True)
    epics_block = "\n".join(f"- {k}" for k in epic_keys) or "(none)"
    stories_block = "\n".join(f"- {k}" for k in story_keys) or "(none)"
    content = f"""# Business Intent

## Has UI
{has_ui}

## Epics
{epics_block}

## Stories
{stories_block}

## Business Idea
{idea}
"""
    BP_BUSINESS_INTENT.write_text(content)
    log(f"blueprint/context/business_intent.md written.")


def _post_assumption_ledger(jira: JiraClient, issue_key: str, assumptions: list) -> None:
    lines = [f"- [{a.get('confidence','?')} confidence] {a.get('description','')}" for a in assumptions]
    try:
        jira.add_comment_adf(
            issue_key,
            make_adf_doc(
                "PO proceeding under the following recorded assumptions (not blocking):\n"
                + "\n".join(lines)
            ),
        )
    except Exception as e:
        log(f"Could not post assumption ledger to {issue_key}: {e}")


def _handle_upstream_hitl(jira: JiraClient, ambiguity: str, state: dict) -> dict:
    log(f"Upstream HITL triggered: {ambiguity}")
    blocking_key = state.get("blocking_issue_key")
    if not blocking_key:
        placeholder = jira.create_issue(
            "HITL: Business Requirements Need Clarification",
            "Story",
            description_adf=make_adf_doc(f"Ambiguity detected:\n{ambiguity}"),
        )
        blocking_key = placeholder["key"]

    JiraFacade(jira).flag_for_human(
        blocking_key,
        make_adf_mention_doc(
            JIRA_HUMAN_ACCOUNT_ID,
            f"Your input is required to resolve this ambiguity: {ambiguity}",
        ),
    )
    log(f"HITL triggered. Blocking: {blocking_key}")
    return {
        **state,
        "status": "AWAITING_HUMAN_REFINEMENT",
        "current_phase": "upstream_hitl",
        "blocking_issue_key": blocking_key,
        "hitl_required": True,
    }
