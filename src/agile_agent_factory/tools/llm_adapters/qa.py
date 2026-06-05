"""LLM adapter for the QA agent — Gherkin criteria + test contract generation."""
from agile_agent_factory.config import QA_MODEL
from agile_agent_factory.tools.llm_client import call_llm_json

QA_SYSTEM = (
    "You are a QA engineer writing Gherkin acceptance criteria and a precise test contract. "
    "Each Gherkin scenario maps 1-to-1 with a pytest test function. "
    "The test_contract must specify the exact test file path, function names, and import paths "
    "so the developer knows exactly what to implement. "
    "Never reference ambiguous imports or shadow the app/ package. "
    "Return JSON only."
)


def build_qa_prompt(summary: str, description_text: str = "") -> str:
    context_block = (
        f"\nStory description for additional context:\n{description_text}\n"
        if description_text
        else ""
    )
    return f"""Write Gherkin acceptance criteria and a structured test contract for this user story.
Return JSON only:
{{
  "acceptance_criteria": [
    "Scenario: <title>\\n  Given <context>\\n  When <action>\\n  Then <outcome>"
  ],
  "test_contract": {{
    "test_file": "tests/test_<feature>.py",
    "test_functions": ["test_<behavior_1>", "test_<behavior_2>"],
    "target_imports": ["from app.<module> import <symbol>"],
    "fixtures": [{{"name": "<fixture_name>", "description": "<what it creates>"}}],
    "sample_data": [{{"<field>": "<value>"}}],
    "edge_cases": ["<edge case description>"]
  }}
}}

Rules:
- test_file must start with "tests/" and end with ".py"
- test_functions must be valid Python identifiers starting with "test_"
- target_imports must use "from app.<module> import <name>" form only
- sample_data is a list of concrete dicts (1–3 examples) usable as pytest parameters
{context_block}
User story: {summary}
"""


def generate_criteria(summary: str, description_text: str, fallback: dict) -> dict:
    return call_llm_json(
        build_qa_prompt(summary, description_text),
        system=QA_SYSTEM,
        fallback=fallback,
        model=QA_MODEL or None,
    )
