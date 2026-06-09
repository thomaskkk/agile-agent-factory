from agile_agent_factory.agents.contract import AgentResult
from agile_agent_factory.config import PRODUCT_ROOT, BP_UX_DECISIONS
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_heading, make_adf_bullet_list, upsert_adf_section
from agile_agent_factory.tools.llm_adapters.ux import generate_ux_spec
from agile_agent_factory.tools.logger import log

BUSINESS_IDEA_PATH = PRODUCT_ROOT / "business_idea.md"

_ALLOWED_UI_TYPES = {"none", "cli", "web", "tui", "desktop", "hybrid"}
_ALLOWED_TECHNOLOGIES = {"argparse", "click", "rich", "Flask", "FastAPI", "Django", "tkinter", "none"}

_UX_FALLBACK: dict = {
    "ui_type": "none",
    "technology": "none",
    "description": "",
    "screens_or_flows": [],
    "state_management": "",
    "design_decisions": [],
}


def _validate_ux_spec(spec: dict) -> dict:
    if spec.get("ui_type") not in _ALLOWED_UI_TYPES:
        log(f"UX agent returned unknown ui_type '{spec.get('ui_type')}' — defaulting to 'none'.")
        return _UX_FALLBACK
    if spec.get("technology") not in _ALLOWED_TECHNOLOGIES:
        log(f"UX agent returned unknown technology '{spec.get('technology')}' — defaulting to 'none'.")
        return _UX_FALLBACK
    return spec


def design_user_experience(
    jira: JiraClient,
    story_keys: list[str],
    gherkin_criteria: dict[str, list[str]],
    state: dict,
    story_feedback: dict[str, str] | None = None,
) -> AgentResult:
    log("Generating UI/UX design specification.")
    business_idea = BUSINESS_IDEA_PATH.read_text() if BUSINESS_IDEA_PATH.exists() else ""

    stories_block = ""
    for key in story_keys:
        issue = jira._request("GET", f"issue/{key}?fields=summary")
        summary = issue["fields"]["summary"]
        criteria = gherkin_criteria.get(key, [])
        stories_block += f"\n**{key}:** {summary}\n"
        if criteria:
            stories_block += "\n".join(f"  - {c.splitlines()[0]}" for c in criteria) + "\n"
        feedback_text = ((story_feedback or {}).get(key) or "").strip()
        if feedback_text:
            stories_block += f"  - Human clarification: {feedback_text}\n"

    raw = generate_ux_spec(business_idea, stories_block, _UX_FALLBACK)
    spec = _validate_ux_spec(raw)

    if spec["ui_type"] != "none":
        _append_ux_to_stories(jira, story_keys, spec)

    _write_ux_decisions(spec)
    log(f"UI/UX design complete: ui_type={spec['ui_type']}, technology={spec['technology']}.")
    return AgentResult(payload={"ux_spec": spec})


def _write_ux_decisions(spec: dict) -> None:
    BP_UX_DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    if spec.get("ui_type", "none") == "none":
        BP_UX_DECISIONS.write_text("# UI/UX\n\nNo user interface required.\n")
        log("blueprint/context/ux_decisions.md written (no-UI marker).")
        return

    flows_md = "\n".join(
        f"- **{f.get('name', '')}** ({f.get('story_key', '')}): {f.get('purpose', '')}"
        for f in spec.get("screens_or_flows", [])
    )
    decisions_md = "\n".join(f"- {d}" for d in spec.get("design_decisions", []))
    content = f"""# UI/UX Decisions

## Type
{spec.get('ui_type')} | Technology: {spec.get('technology')}

## Description
{spec.get('description', '')}

## Screens / Flows
{flows_md}

## State Management
{spec.get('state_management', '')}

## Design Decisions
{decisions_md}
"""
    BP_UX_DECISIONS.write_text(content)
    log("blueprint/context/ux_decisions.md written.")


def _append_ux_to_stories(jira: JiraClient, story_keys: list[str], spec: dict) -> None:
    decisions = spec.get("design_decisions", [])
    flows = spec.get("screens_or_flows", [])
    flows_by_story: dict[str, list[str]] = {}
    for flow in flows:
        key = flow.get("story_key", "")
        label = f"{flow.get('name', '')}: {flow.get('purpose', '')}"
        flows_by_story.setdefault(key, []).append(label)

    for story_key in story_keys:
        issue = jira._request("GET", f"issue/{story_key}?fields=description")
        existing = issue["fields"].get("description") or {}
        ux_nodes = [
            make_adf_heading("UI/UX Design"),
            {"type": "paragraph", "content": [
                {"type": "text", "text": f"Type: {spec['ui_type']} | Technology: {spec['technology']}"},
            ]},
        ]
        story_flows = flows_by_story.get(story_key, [])
        if story_flows:
            ux_nodes.append(make_adf_heading("Flows", level=4))
            ux_nodes.append(make_adf_bullet_list(story_flows))
        if decisions:
            ux_nodes.append(make_adf_heading("Design Decisions", level=4))
            ux_nodes.append(make_adf_bullet_list(decisions))

        new_description = upsert_adf_section(existing, "UI/UX Design", ux_nodes)
        jira.update_issue_description(story_key, new_description)
        log(f"Appended UI/UX design to {story_key}.")
