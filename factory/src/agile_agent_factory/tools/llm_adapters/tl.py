"""LLM adapter for the Tech Lead agent — architecture/file-contract design."""
import json

from agile_agent_factory.config import TL_MODEL
from agile_agent_factory.tools.llm_client import call_llm_json

TL_SYSTEM = (
    "You are a Tech Lead designing a Python software architecture. "
    "All app code goes in app/, all tests in tests/. "
    "Never produce nested app/app/ or tests/tests/ paths. "
    "All imports must use `from app.<module> import <name>`. "
    "Return JSON only."
)


def _ux_block(ux_spec: dict) -> str:
    if ux_spec and ux_spec.get("ui_type", "none") != "none":
        decisions = "\n".join(f"- {d}" for d in ux_spec.get("design_decisions", []))
        flows = "\n".join(
            f"- {f.get('name', '')}: {f.get('purpose', '')}"
            for f in ux_spec.get("screens_or_flows", [])
        )
        return f"""
UI/UX Design Specification:
  Type: {ux_spec.get('ui_type')} | Technology: {ux_spec.get('technology')}
  Description: {ux_spec.get('description', '')}
  Flows:
{flows}
  Design Decisions:
{decisions}
  State Management: {ux_spec.get('state_management', '')}
"""
    return ""


def build_tl_prompt(idea: str, story_keys: list[str], ux_spec: dict, ready_contracts: dict) -> str:
    ready_contracts_block = json.dumps(
        {key: ready_contracts.get(key, {}) for key in story_keys},
        indent=2,
        sort_keys=True,
    )
    ux_block = _ux_block(ux_spec)
    return f"""Design the software architecture for this product.
Return JSON only:
{{
  "files": [
    {{"path": "app/module.py", "purpose": "what it does", "functions": ["sig(args) -> type"]}}
  ],
  "subtasks": [
    {{"title": "Implement X", "story_key": "<key>", "description": "details"}}
  ],
  "import_rules": "...",
  "test_command": "uv run pytest tests/ -v"
}}

Story keys: {story_keys}
Validated Definition-of-Ready contracts (authoritative):
{ready_contracts_block}

Business idea:
{idea}
{ux_block}"""


def generate_architecture(idea: str, story_keys: list[str], ux_spec: dict, ready_contracts: dict, fallback: dict) -> dict:
    return call_llm_json(
        build_tl_prompt(idea, story_keys, ux_spec, ready_contracts),
        system=TL_SYSTEM,
        fallback=fallback,
        model=TL_MODEL or None,
    )
