import json
import re

from agile_agent_factory.config import (
    PRODUCT_ROOT,
    BP_ARCH_DECISIONS, BP_ARCH_CONSTRAINTS, bp_task_path,
)
from agile_agent_factory.tools.dependencies import UX_TECH_TO_PACKAGE
from agile_agent_factory.tools.jira_client import JiraClient, make_adf_doc
from agile_agent_factory.tools.llm_adapters.tl import generate_architecture
from agile_agent_factory.tools.logger import log
from agile_agent_factory.tools.workflow import WorkflowState

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


def _transition_all(jira: JiraClient, keys: list[str], target: WorkflowState) -> None:
    for key in keys:
        try:
            jira.transition_to(key, target)
        except ValueError as e:
            log(str(e))


def design_architecture(
    jira: JiraClient,
    story_keys: list[str],
    gherkin_criteria: dict[str, list[str]],
    ux_spec: dict,
    state: dict,
    ready_contracts: dict[str, dict] | None = None,
) -> dict:
    all_upstream_keys = story_keys + state.get("epic_keys", [])
    _transition_all(jira, all_upstream_keys, WorkflowState.TECH_REFINEMENT)

    idea = BUSINESS_IDEA_PATH.read_text()

    ready_contracts = ready_contracts or state.get("ready_contracts", {}) or {}

    if state.get("current_phase") == "upstream_arch_done" and state.get("architecture"):
        log("Architecture already generated — reusing stored result (no LLM call).")
        result = state["architecture"]
    else:
        result = generate_architecture(idea, story_keys, ux_spec, ready_contracts, _ARCH_FALLBACK)

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

    _write_architecture_files(result)
    for sk in story_keys:
        _write_story_task(sk, result, gherkin_criteria, ux_spec, ready_contracts.get(sk, {}))
    _transition_all(jira, all_upstream_keys, WorkflowState.TO_DEVELOPMENT)
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



def _write_architecture_files(arch: dict) -> None:
    BP_ARCH_DECISIONS.parent.mkdir(parents=True, exist_ok=True)

    files_section = "\n".join(
        f"### `{f['path']}`\n**Purpose:** {f.get('purpose', '')}\n**Functions:** {', '.join(f.get('functions', []))}"
        for f in arch.get("files", [])
    )
    deps = [d for d in arch.get("dependencies", []) if isinstance(d, str) and d.strip()]
    deps_section = "\n".join(f"- `{d}`" for d in deps) if deps else "(none)"
    BP_ARCH_DECISIONS.write_text(
        f"# Architecture Decisions\n\n## File Contracts\n\n{files_section}\n\n## Third-Party Dependencies\n{deps_section}\n"
    )

    BP_ARCH_CONSTRAINTS.parent.mkdir(parents=True, exist_ok=True)
    expected_files = "\n".join(f"- `{f['path']}`" for f in arch.get("files", []))
    BP_ARCH_CONSTRAINTS.write_text(
        f"""# Architecture Constraints

## Import Rules
{arch.get('import_rules', 'Use `from app.<module> import <name>` from the product root.')}

## Permitted Target Directories
- `../app/` — all production source code
- `../tests/` — all test code

## Test Command
```bash
{arch.get('test_command', 'uv run pytest ../tests/ -v')}
```

## Expected Output Files
{expected_files}

## Definition of Done
- All pytest tests pass with exit code 0
- No absolute paths in any generated file
- All imports use `app.*` from the product root
- No nested `app/app/` or `tests/tests/` directories
"""
    )
    log("blueprint/architecture/decisions.md and constraints.md written.")


def _write_story_task(
    story_key: str,
    arch: dict,
    gherkin_criteria: dict[str, list[str]],
    ux_spec: dict,
    ready_contract: dict | None = None,
) -> None:
    path = bp_task_path(story_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    criteria = gherkin_criteria.get(story_key, [])
    criteria_block = "\n\n".join(criteria) if criteria else "(no criteria)"

    files_section = "\n".join(
        f"- `{f['path']}`: {f.get('purpose', '')} | Functions: {', '.join(f.get('functions', []))}"
        for f in arch.get("files", [])
    )

    ux_flows_section = ""
    if ux_spec and ux_spec.get("ui_type", "none") != "none":
        relevant = [
            f for f in ux_spec.get("screens_or_flows", [])
            if f.get("story_key") == story_key
        ]
        if relevant:
            flows_md = "\n".join(
                f"- **{f.get('name', '')}**: {f.get('purpose', '')}"
                for f in relevant
            )
            ux_flows_section = f"\n## UX Flows (this story)\n{flows_md}\n"

    ready_contract_block = json.dumps(ready_contract or {}, indent=2, sort_keys=True)

    tc = (ready_contract or {}).get("test_contract", {})
    test_contract_section = ""
    if tc:
        tc_test_file = tc.get("test_file", "")
        tc_functions = tc.get("test_functions", [])
        tc_imports = tc.get("target_imports", [])
        tc_fixtures = tc.get("fixtures", [])
        tc_sample_data = tc.get("sample_data", [])
        tc_edge_cases = tc.get("edge_cases", [])

        functions_block = "\n".join(f"- `{f}`" for f in tc_functions) or "(none)"
        imports_block = "\n".join(f"- `{i}`" for i in tc_imports) or "(none)"
        fixtures_block = (
            "\n".join(f"- `{fix['name']}`: {fix.get('description', '')}" for fix in tc_fixtures if isinstance(fix, dict))
            or "(none)"
        )
        sample_block = (
            "\n".join(f"- `{json.dumps(d)}`" for d in tc_sample_data[:3])
            or "(none)"
        )
        edge_block = "\n".join(f"- {e}" for e in tc_edge_cases) or "(none)"

        # Derive allowed app file paths from target_imports
        _allowed_app_files = []
        for imp in tc_imports:
            m = re.match(r"from (app(?:\.\w+)+) import", imp)
            if m:
                path_str = m.group(1).replace(".", "/") + ".py"
                if path_str not in _allowed_app_files:
                    _allowed_app_files.append(path_str)

        _write_scope_lines = []
        if tc_test_file:
            _write_scope_lines.append(f"- `{tc_test_file}` (test file — create or overwrite)")
        for p in _allowed_app_files:
            _write_scope_lines.append(f"- `{p}` (derived from target imports)")
        write_scope_block = "\n".join(_write_scope_lines) if _write_scope_lines else "- (no specific scope — follow architecture file contracts)"

        test_contract_section = f"""
## Test Contract
**Test file:** `{tc_test_file}`

### Expected Test Functions (implement these exactly)
{functions_block}

### Target Imports (these MUST be importable after your implementation)
{imports_block}

### Fixtures
{fixtures_block}

### Sample Data
{sample_block}

### Edge Cases to Cover
{edge_block}

## Write Scope — Strictly Enforced
You MUST write ONLY to the files listed below. Writing to any other file is FORBIDDEN.

{write_scope_block}

Do NOT touch `bootstrap.py`, routes, services, database singletons, shared infrastructure,
or any file owned by another story. Modifying files outside this scope breaks existing
passing tests and will trigger HITL intervention.
"""

    content = f"""# Task: {story_key}

## Validated Definition of Ready Contract
```json
{ready_contract_block}
```

## Acceptance Criteria
{criteria_block}
{ux_flows_section}{test_contract_section}
## Read-Only Shared Architecture Context

### File Contracts
{files_section}

### Import Rules
{arch.get('import_rules', 'Use `from app.<module> import <name>` from the product root.')}

### Test Command
```bash
{arch.get('test_command', 'uv run pytest ../tests/ -v')}
```

### Definition of Done
- All pytest tests pass with exit code 0
- No absolute paths in any generated file
- All imports use `app.*` from the product root
- No nested `app/app/` or `tests/tests/` directories

## Context Pointers
- Business intent: `blueprint/context/business_intent.md`
- Full UX spec: `blueprint/context/ux_decisions.md`
- QA criteria (this story): `blueprint/context/qa_criteria/{story_key}.md`
- Architecture decisions: `blueprint/architecture/decisions.md`
- Architecture constraints: `blueprint/architecture/constraints.md`
"""
    path.write_text(content)
    log(f"blueprint/tasks/{story_key}.md written.")
