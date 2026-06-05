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
    "ambiguity_description": "",
    "has_ui": False,
    "epics": [],
}


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

    if result.get("has_ambiguity"):
        return AgentResult(payload=_handle_upstream_hitl(jira, result.get("ambiguity_description", "Unknown ambiguity"), state))

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
    return AgentResult(payload={
        **state,
        "current_phase": "upstream_po_done",
        "epic_keys": epic_keys,
        "story_keys": story_keys,
        "has_ui": result.get("has_ui", False),
        "story_to_epic": story_to_epic,
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


def _handle_upstream_hitl(jira: JiraClient, ambiguity: str, state: dict) -> dict:
    log(f"Upstream HITL triggered: {ambiguity}")
    placeholder = jira.create_issue(
        "HITL: Business Requirements Need Clarification",
        "Story",
        description_adf=make_adf_doc(f"Ambiguity detected:\n{ambiguity}"),
    )
    JiraFacade(jira).flag_for_human(
        placeholder["key"],
        make_adf_mention_doc(
            JIRA_HUMAN_ACCOUNT_ID,
            f"Your input is required to resolve this ambiguity: {ambiguity}",
        ),
    )
    log(f"HITL triggered. Blocking: {placeholder['key']}")
    return {
        **state,
        "status": "AWAITING_HUMAN_REFINEMENT",
        "current_phase": "upstream_hitl",
        "blocking_issue_key": placeholder["key"],
        "hitl_required": True,
    }
