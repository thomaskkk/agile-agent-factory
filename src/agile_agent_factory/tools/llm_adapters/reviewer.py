"""LLM adapter for the Reviewer agent — DoD audit of generated code."""
from agile_agent_factory.config import REVIEWER_MODEL
from agile_agent_factory.tools.llm_client import call_llm_json

REVIEW_SYSTEM = (
    "You are a Code Reviewer auditing generated code (Python, HTML templates, CSS, JS) "
    "against a Definition of Done. "
    "IMPORTANT: For Python ASGI/WSGI applications (FastAPI, Flask, Starlette), using "
    "Starlette TestClient or httpx AsyncClient to exercise HTTP endpoints IS the correct "
    "testing approach — it does not require binding to a real network socket. A server "
    "wrapper whose start()/stop() methods manage lifecycle state and whose underlying ASGI "
    "app passes all HTTP tests via TestClient fully satisfies 'listens for connections' and "
    "'accepts HTTP requests' acceptance criteria. "
    "Output ONLY a JSON object — no prose, no preamble, no markdown fences. "
    "Schema: {\"approved\": bool, \"rejection_reason\": \"\"}"
)


def build_review_prompt(dod_section: str, files_block: str, write_scope: list[str] | None = None) -> str:
    scope_instruction = ""
    if write_scope:
        owned = "\n".join(f"  - {f}" for f in write_scope)
        scope_instruction = f"""
Story write scope — files owned by this story:
{owned}

Evaluate correctness and DoD compliance ONLY for the files listed above.
Issues found in files OUTSIDE this scope are pre-existing problems owned by other
stories and MUST NOT cause this review to fail — note them as informational only.
"""

    return f"""Review the generated code against the blueprint DoD.
You are shown the COMPLETE contents of all generated files below — Python source,
HTML templates, CSS, and JS. Do not assume any file type is missing unless it is
absent from the list. Judge functional correctness and DoD coverage only.
{scope_instruction}
Acceptance criteria (DoD):
{dod_section[:8000]}

Generated files:
{files_block}

Output ONLY valid JSON, nothing else:
{{"approved": true, "rejection_reason": ""}}
"""


def generate_review(dod_section: str, files_block: str, write_scope: list[str] | None, fallback: dict) -> dict:
    return call_llm_json(
        build_review_prompt(dod_section, files_block, write_scope),
        system=REVIEW_SYSTEM,
        fallback=fallback,
        model=REVIEWER_MODEL or None,
    )
