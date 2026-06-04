from agile_agent_factory.config import DRY_RUN, QA_MODEL, bp_qa_criteria_path
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_heading, make_adf_bullet_list
from agile_agent_factory.tools.llm_client import call_llm_json
from agile_agent_factory.tools.logger import log

_GHERKIN_FALLBACK = {"acceptance_criteria": []}


def inject_gherkin_criteria(jira: JiraClient, story_keys: list[str]) -> dict[str, list[str]]:
    all_criteria: dict[str, list[str]] = {}

    for story_key in story_keys:
        issue = jira._request("GET", f"issue/{story_key}?fields=summary,description")
        summary = issue["fields"]["summary"]
        existing_description = issue["fields"].get("description") or {}
        log(f"Generating Gherkin criteria for {story_key}: {summary}")

        system = (
            "You are a QA engineer writing Gherkin acceptance criteria. "
            "Each scenario must map 1-to-1 with a pytest test function using Given/When/Then. "
            "Never reference ambiguous imports or shadow the app/ package. "
            "Return JSON only."
        )
        prompt = f"""Write Gherkin acceptance criteria for this user story.
Return JSON only:
{{
  "acceptance_criteria": [
    "Scenario: <title>\\n  Given <context>\\n  When <action>\\n  Then <outcome>"
  ]
}}

User story: {summary}
"""
        result = call_llm_json(prompt, system=system, fallback=_GHERKIN_FALLBACK, model=QA_MODEL or None)
        criteria = result.get("acceptance_criteria", [])
        all_criteria[story_key] = criteria
        _write_qa_criteria(story_key, criteria)

        if criteria:
            existing_content = existing_description.get("content", []) if existing_description else []
            criteria_nodes = [make_adf_heading("Acceptance Criteria"), make_adf_bullet_list(criteria)]
            new_description = {"version": 1, "type": "doc", "content": existing_content + criteria_nodes}
            jira.update_issue_description(story_key, new_description)
            try:
                jira.transition_issue(story_key, "To Tech Refinement")
            except ValueError as e:
                log(str(e))

    return all_criteria


def _write_qa_criteria(story_key: str, criteria: list[str]) -> None:
    path = bp_qa_criteria_path(story_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n\n".join(criteria) if criteria else "(no criteria generated)"
    content = f"# QA Acceptance Criteria — {story_key}\n\n{blocks}\n"
    path.write_text(content)
    log(f"blueprint/context/qa_criteria/{story_key}.md written.")
