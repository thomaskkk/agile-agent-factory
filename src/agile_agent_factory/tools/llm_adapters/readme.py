"""LLM adapter for the README agent — user-facing README.md generation.

Unlike the other adapters this calls ``call_llm`` (free-text), not
``call_llm_json``.
"""
from agile_agent_factory.config import README_MODEL
from agile_agent_factory.tools.llm_client import call_llm

README_SYSTEM = (
    "You are a technical writer producing a user-facing README.md for a Python software product. "
    "Write clear Markdown. Do not mention Jira, agile, epics, stories, "
    "Gherkin, or any internal factory tooling. "
    "Do not include fenced code blocks for the overall document — "
    "only use fenced blocks for code examples inside the README."
)


def build_readme_prompt(business_idea: str, blueprint: str, pyproject: str, files_block: str) -> str:
    return f"""You are writing a README.md for a software product.
Your primary goal: make the person who wrote the original business requirements happy.
They should be able to read this README and immediately know how to use the product.

Rules:
- MANDATORY first section: "## Setup" — it MUST appear before any run or test commands.
  Examine the pyproject.toml below to determine what dependencies are needed and which
  package manager is expected, then write the Setup section as follows:
    - If pyproject.toml has a [dependency-groups] table (uv convention): provide two paths:
        Option A (uv): `uv sync` then prefix commands with `uv run`
        Option B (plain pip): `pip install <each runtime dep>` — list packages explicitly,
        do NOT say `pip install .` because there is no [project] table to install from.
    - If pyproject.toml has [project].dependencies (standard): provide two paths:
        Option A (uv): `uv sync`
        Option B (plain pip): `pip install .`
    - Always state the Python version requirement (3.10+ unless the code shows otherwise).
- Lead with HOW TO USE after Setup — whatever the business idea says the user will do
  (run a script, import a library, call a CLI, etc.) must follow immediately after Setup.
- Never omit the Setup section even for trivial apps — missing setup instructions are the
  most common reason users get "No module named X" errors.
- Infer usage from the actual source files provided. Do NOT fabricate commands or APIs
  that don't exist in the code.
- In the "Running the Tests" section: do NOT repeat the dependency install step — just
  reference "After completing Setup above, also install pytest" and show the run command.
- Keep sections minimal — only include what actually exists in the product.
- Do not mention Jira, epics, stories, Gherkin, or any internal factory tooling.
- Do not include fenced code blocks for the overall document.

Original business requirements (what the user asked for):
{business_idea}

Architecture blueprint (what was planned and built):
{blueprint[:8000]}

pyproject.toml (use this to determine dependencies and package manager):
```toml
{pyproject}
```

Generated source files (what was actually implemented):
{files_block if files_block else "(no files found)"}

Write the README now.
"""


def generate(business_idea: str, blueprint: str, pyproject: str, files_block: str) -> str:
    return call_llm(
        build_readme_prompt(business_idea, blueprint, pyproject, files_block),
        system=README_SYSTEM,
        model=README_MODEL or None,
    )
