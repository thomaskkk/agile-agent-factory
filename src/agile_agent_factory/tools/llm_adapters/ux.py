"""LLM adapter for the UX agent — interaction/flow spec generation."""
from agile_agent_factory.config import UX_MODEL
from agile_agent_factory.tools.llm_client import call_llm_json

UX_SYSTEM = (
    "You are a UX designer embedded in an Agile team. "
    "Your output defines interaction intent and user flow — never raw code or framework markup. "
    "Choose exactly one technology from the allowed list; do not invent new ones. "
    "Respond ONLY with valid JSON matching the exact schema."
)


def build_ux_prompt(business_idea: str, stories_block: str) -> str:
    return f"""Design the user experience for this product.
Return JSON only:
{{
  "ui_type": "cli",
  "technology": "argparse",
  "description": "High-level interaction model — intent and flow only, no code",
  "screens_or_flows": [
    {{
      "name": "screen or command name",
      "purpose": "what the user does here",
      "key_elements": ["element or flag"],
      "story_key": "STORY-KEY"
    }}
  ],
  "state_management": "How data flows and state is managed at runtime",
  "design_decisions": ["Decision 1", "Decision 2"]
}}

Allowed ui_type values: none, cli, web, tui, desktop, hybrid
Allowed technology values: argparse, click, rich, Flask, FastAPI, Django, tkinter, none

Rules:
- Output interaction intent only — no code snippets, no HTML, no component implementations.
- Pick the technology that best matches the business idea from the allowed list.
- If the product has no user-facing interface, use ui_type "none" and technology "none".

Business idea:
{business_idea}

User stories and acceptance criteria:
{stories_block}
"""


def generate_ux_spec(business_idea: str, stories_block: str, fallback: dict) -> dict:
    return call_llm_json(
        build_ux_prompt(business_idea, stories_block),
        system=UX_SYSTEM,
        fallback=fallback,
        model=UX_MODEL or None,
    )
