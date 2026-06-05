"""LLM adapter for the PO agent — epic/story decomposition + ambiguity check."""
from agile_agent_factory.config import PO_MODEL
from agile_agent_factory.tools.llm_client import call_llm_json

PO_SYSTEM = (
    "You are a Product Owner in an Agile team. "
    "Analyze the business requirements for logical contradictions, missing mandatory data, "
    "or unfeasible scope. Then define Epics and User Stories with a Definition of Done. "
    "Set has_ui to true if the product needs any user-facing interface (CLI with commands, "
    "web UI, TUI, desktop app). Set it to false for pure library modules imported by other code. "
    "Respond ONLY with valid JSON matching the exact schema below."
)


def build_po_prompt(idea: str, hitl_feedback: str = "") -> str:
    feedback_block = (
        f"\nHuman clarification already provided for a prior ambiguity check:\n{hitl_feedback}\n"
        "Treat the above as resolved context. Only flag genuinely NEW ambiguities "
        "not addressed by the clarification above.\n"
        if hitl_feedback
        else ""
    )
    return f"""Return JSON only — no extra text:
{{
  "has_ambiguity": false,
  "ambiguity_description": "",
  "has_ui": false,
  "epics": [
    {{
      "title": "Epic title",
      "description": "Epic description",
      "stories": [
        {{
          "title": "Story title",
          "description": "As a user, I want...",
          "definition_of_done": ["criterion 1", "criterion 2"]
        }}
      ]
    }}
  ]
}}
{feedback_block}
Business idea to analyze:
{idea}
"""


def analyze_business(idea: str, hitl_feedback: str, fallback: dict) -> dict:
    return call_llm_json(
        build_po_prompt(idea, hitl_feedback),
        system=PO_SYSTEM,
        fallback=fallback,
        model=PO_MODEL or None,
    )
