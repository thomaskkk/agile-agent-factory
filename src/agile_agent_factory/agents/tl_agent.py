from agile_agent_factory.config import PRODUCT_ROOT, BLUEPRINT_PATH, TL_MODEL
from agile_agent_factory.tools.dependencies import UX_TECH_TO_PACKAGE
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc
from agile_agent_factory.tools.llm_client import call_llm_json
from agile_agent_factory.tools.logger import log

BUSINESS_IDEA_PATH = PRODUCT_ROOT / "business_idea.md"

_ARCH_FALLBACK = {
    "files": [
        {"path": "app/__init__.py", "purpose": "Package init", "functions": []},
        {"path": "app/cli.py", "purpose": "CLI entry point", "functions": ["main()"]},
        {"path": "app/tracker.py", "purpose": "Core task logic", "functions": ["add_task(title: str) -> dict", "list_tasks() -> list", "complete_task(task_id: int) -> bool"]},
        {"path": "tests/__init__.py", "purpose": "Test package init", "functions": []},
        {"path": "tests/test_tracker.py", "purpose": "Unit tests", "functions": []},
    ],
    "subtasks": [],
    "import_rules": "All imports use `from app.<module> import <name>` from the product root.",
    "test_command": "uv run pytest ../tests/ -v",
    "dependencies": [],
}


def _transition_all(jira: JiraClient, keys: list[str], target: str) -> None:
    for key in keys:
        try:
            jira.transition_issue(key, target)
        except ValueError as e:
            log(str(e))


def design_architecture(
    jira: JiraClient,
    story_keys: list[str],
    gherkin_criteria: dict[str, list[str]],
    ux_spec: dict,
    state: dict,
) -> dict:
    all_upstream_keys = story_keys + state.get("epic_keys", [])
    _transition_all(jira, all_upstream_keys, "Tech Refinement")

    idea = BUSINESS_IDEA_PATH.read_text()

    system = (
        "You are a Tech Lead designing a Python software architecture. "
        "All app code goes in app/, all tests in tests/. "
        "Never produce nested app/app/ or tests/tests/ paths. "
        "All imports must use `from app.<module> import <name>`. "
        "Return JSON only."
    )
    ux_block = ""
    if ux_spec and ux_spec.get("ui_type", "none") != "none":
        decisions = "\n".join(f"- {d}" for d in ux_spec.get("design_decisions", []))
        flows = "\n".join(
            f"- {f.get('name', '')}: {f.get('purpose', '')}"
            for f in ux_spec.get("screens_or_flows", [])
        )
        ux_block = f"""
UI/UX Design Specification:
  Type: {ux_spec.get('ui_type')} | Technology: {ux_spec.get('technology')}
  Description: {ux_spec.get('description', '')}
  Flows:
{flows}
  Design Decisions:
{decisions}
  State Management: {ux_spec.get('state_management', '')}
"""

    prompt = f"""Design the software architecture for this product.
Return JSON only:
{{
  "files": [
    {{"path": "app/module.py", "purpose": "what it does", "functions": ["sig(args) -> type"]}}
  ],
  "subtasks": [
    {{"title": "Implement X", "story_key": "<key>", "description": "details"}}
  ],
  "import_rules": "...",
  "test_command": "uv run pytest ../tests/ -v"
}}

Story keys: {story_keys}
Business idea:
{idea}
{ux_block}"""
    if state.get("current_phase") == "upstream_arch_done" and state.get("architecture"):
        log("Architecture already generated — reusing stored result (no LLM call).")
        result = state["architecture"]
    else:
        result = call_llm_json(prompt, system=system, fallback=_ARCH_FALLBACK, model=TL_MODEL or None)

    subtask_type = jira.get_subtask_issue_type()
    subtask_keys: dict[str, str] = dict(state.get("subtasks", {}))
    for subtask in result.get("subtasks", []):
        if subtask["title"] in subtask_keys:
            log(f"Subtask already created — skipping: {subtask['title']}")
            continue
        parent_key = subtask.get("story_key") or (story_keys[0] if story_keys else None)
        if parent_key:
            task = jira.create_issue(
                subtask["title"],
                subtask_type,
                description_adf=make_adf_doc(subtask.get("description", "")),
                parent_key=parent_key,
            )
            subtask_keys[subtask["title"]] = task["key"]
            log(f"Created Subtask: {task['key']} — {subtask['title']}")

    _write_handoff_blueprint(idea, story_keys, gherkin_criteria, result, ux_spec)
    _transition_all(jira, all_upstream_keys, "To Development")
    dependencies = [d for d in result.get("dependencies", []) if isinstance(d, str) and d.strip()]
    # The LLM sometimes omits dependencies entirely — seed the UX technology so an
    # obvious framework (e.g. Flask) is never lost. The downstream test-time scan is
    # the ultimate safety net, but this keeps state visible and correct upfront.
    tech = (ux_spec or {}).get("technology", "").lower()
    if tech in UX_TECH_TO_PACKAGE and UX_TECH_TO_PACKAGE[tech] not in dependencies:
        dependencies.append(UX_TECH_TO_PACKAGE[tech])
    return {
        **state,
        "current_phase": "upstream_tl_done",
        "subtasks": subtask_keys,
        "dependencies": dependencies,
        "architecture": result,
    }


def _write_handoff_blueprint(
    idea: str,
    story_keys: list[str],
    gherkin_criteria: dict[str, list[str]],
    arch: dict,
    ux_spec: dict,
) -> None:
    log(f"Writing handoff_blueprint.md.")

    files_section = "\n".join(
        f"- `{f['path']}`: {f.get('purpose', '')} | Functions: {', '.join(f.get('functions', []))}"
        for f in arch.get("files", [])
    )
    criteria_section = "\n\n".join(
        f"**{key}:**\n" + "\n".join(gherkin_criteria.get(key, ["(no criteria)"]))
        for key in story_keys
    )
    expected_files = "\n".join(f"- `{f['path']}`" for f in arch.get("files", []))

    deps = [d for d in arch.get("dependencies", []) if isinstance(d, str) and d.strip()]
    deps_section = ""
    if deps:
        deps_section = "\n## Third-Party Dependencies\n" + "\n".join(f"- `{d}`" for d in deps) + "\n"

    ux_section = ""
    if ux_spec and ux_spec.get("ui_type", "none") != "none":
        decisions_md = "\n".join(f"- {d}" for d in ux_spec.get("design_decisions", []))
        flows_md = "\n".join(
            f"- **{f.get('name', '')}** ({f.get('story_key', '')}): {f.get('purpose', '')}"
            for f in ux_spec.get("screens_or_flows", [])
        )
        ux_section = f"""
## UI/UX Design Specification
**Type:** {ux_spec.get('ui_type')} | **Technology:** {ux_spec.get('technology')}
**Description:** {ux_spec.get('description', '')}

### Screens / Flows
{flows_md}

### State Management
{ux_spec.get('state_management', '')}

### Design Decisions
{decisions_md}
"""

    content = f"""# Handoff Blueprint

## Product Summary
{idea[:800]}

## User Stories
{chr(10).join(f'- {k}' for k in story_keys)}

## Gherkin Acceptance Criteria
{criteria_section}
{ux_section}
## Architecture: File Contracts
{files_section}

## Import Rules
{arch.get('import_rules', 'Use `from app.<module> import <name>` from the product root.')}

## Permitted Target Directories
- `../app/` — all production source code
- `../tests/` — all test code

## Test Command
```bash
{arch.get('test_command', 'uv run pytest ../tests/ -v')}
```
{deps_section}

## Python Import Rules
All imports resolve from the product root: `from app.<module> import <name>`.
PYTHONPATH is set to the product root before running pytest.

## Expected Output Files
{expected_files}

## Definition of Done
- All pytest tests pass with exit code 0
- No absolute paths in any generated file
- All imports use `app.*` from the product root
- No nested `app/app/` or `tests/tests/` directories
"""
    BLUEPRINT_PATH.write_text(content)
    log(f"handoff_blueprint.md written to {BLUEPRINT_PATH}.")
