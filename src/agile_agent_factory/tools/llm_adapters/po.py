"""LLM adapter for the PO agent — epic/story decomposition + ambiguity check."""
from agile_agent_factory.config import PO_MODEL
from agile_agent_factory.tools.llm_client import call_llm_json

PO_SYSTEM = (
    "You are a Product Owner in an Agile team. "
    "Analyze the business requirements. "
    "Set has_ambiguity=true and hitl_required=true ONLY for blocking gaps: logical contradictions, "
    "missing mandatory data with no safe default, or infeasible scope. "
    "For non-critical ambiguity (unclear but can proceed with a reasonable assumption), "
    "record the assumption in the 'assumptions' list with numeric 'confidence' (0.0–1.0, where 1.0 "
    "= certain) and numeric 'impact' (0.0–1.0, where 1.0 = critical to product success), "
    "then leave has_ambiguity=false, hitl_required=false. "
    "Before including any assumption, apply the self-resolve checklist: "
    "for each assumption where impact * (1 - confidence) > 0.5 (high-risk), re-read the business "
    "idea text for evidence that resolves it. If evidence is found, raise confidence accordingly. "
    "Only include assumptions in the output where self-resolution was attempted and failed. "
    "Then define Epics and User Stories with a Definition of Done. "
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
  "hitl_required": false,
  "ambiguity_description": "",
  "assumptions": [
    {{
      "description": "Assumption text",
      "confidence": 0.8,
      "impact": 0.6,
      "rationale": "Why this assumption was made"
    }}
  ],
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
