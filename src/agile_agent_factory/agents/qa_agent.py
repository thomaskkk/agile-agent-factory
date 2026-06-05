from agile_agent_factory.config import DRY_RUN, bp_qa_criteria_path
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_heading, make_adf_bullet_list
from agile_agent_factory.tools.llm_adapters.qa import generate_criteria
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState

_QA_FALLBACK = {"acceptance_criteria": [], "test_contract": {}}


def inject_gherkin_criteria(
    jira: JiraClient, story_keys: list[str]
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Return (criteria_dict, test_contracts_dict) for the given stories."""
    all_criteria: dict[str, list[str]] = {}
    all_test_contracts: dict[str, dict] = {}

    for story_key in story_keys:
        issue = jira._request("GET", f"issue/{story_key}?fields=summary,description")
        summary = issue["fields"]["summary"]
        existing_description = issue["fields"].get("description") or {}
        log(f"Generating Gherkin criteria for {story_key}: {summary}")

        result = generate_criteria(summary, "", _QA_FALLBACK)
        criteria = result.get("acceptance_criteria", [])
        test_contract = result.get("test_contract") or {}

        all_criteria[story_key] = criteria
        all_test_contracts[story_key] = test_contract
        _write_qa_criteria(story_key, criteria, test_contract)

        if criteria:
            existing_content = existing_description.get("content", []) if existing_description else []
            criteria_nodes = [make_adf_heading("Acceptance Criteria"), make_adf_bullet_list(criteria)]
            new_description = {"version": 1, "type": "doc", "content": existing_content + criteria_nodes}
            jira.update_issue_description(story_key, new_description)
            try:
                jira.transition_to(story_key, WorkflowState.TECH_REFINEMENT)
            except ValueError as e:
                log(str(e))

    return all_criteria, all_test_contracts


def _write_qa_criteria(story_key: str, criteria: list[str], test_contract: dict) -> None:
    path = bp_qa_criteria_path(story_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n\n".join(criteria) if criteria else "(no criteria generated)"

    tc_lines = ""
    if test_contract:
        tc_lines = (
            "\n\n## Test Contract\n"
            f"- **Test file:** `{test_contract.get('test_file', '')}`\n"
            f"- **Test functions:** {', '.join(f'`{f}`' for f in test_contract.get('test_functions', []))}\n"
            f"- **Target imports:** {', '.join(f'`{i}`' for i in test_contract.get('target_imports', []))}\n"
            f"- **Edge cases:** {', '.join(test_contract.get('edge_cases', []))}\n"
        )

    content = f"# QA Acceptance Criteria — {story_key}\n\n{blocks}\n{tc_lines}"
    path.write_text(content)
    log(f"blueprint/context/qa_criteria/{story_key}.md written.")
