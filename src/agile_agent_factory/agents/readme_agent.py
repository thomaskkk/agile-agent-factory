from agile_agent_factory.config import PRODUCT_ROOT, BLUEPRINT_PATH, README_MODEL
from agile_agent_factory.tools.llm_client import call_llm
from agile_agent_factory.tools.logger import log

BUSINESS_IDEA_PATH = PRODUCT_ROOT / "business_idea.md"
README_PATH = PRODUCT_ROOT / "README.md"
PYPROJECT_PATH = PRODUCT_ROOT / "pyproject.toml"

MAX_README_FILE_CHARS = 2000
MAX_README_FILES = 15
MAX_README_TOTAL_CHARS = 20000


def generate_readme(state: dict) -> None:
    log("Generating README.md for product output.")

    blueprint = BLUEPRINT_PATH.read_text() if BLUEPRINT_PATH.exists() else ""
    business_idea = BUSINESS_IDEA_PATH.read_text() if BUSINESS_IDEA_PATH.exists() else ""
    pyproject = PYPROJECT_PATH.read_text() if PYPROJECT_PATH.exists() else ""

    file_contents: dict[str, str] = {}
    for target_dir in ("app", "tests"):
        d = PRODUCT_ROOT / target_dir
        if d.exists():
            for f in sorted(d.rglob("*.py")):
                rel = str(f.relative_to(PRODUCT_ROOT))
                file_contents[rel] = f.read_text()

    files_block = "\n\n".join(
        f"### {path}\n```python\n{content[:MAX_README_FILE_CHARS]}\n```"
        for path, content in list(file_contents.items())[:MAX_README_FILES]
    )[:MAX_README_TOTAL_CHARS]

    system = (
        "You are a technical writer producing a user-facing README.md for a Python software product. "
        "Write clear Markdown. Do not mention Jira, agile, epics, stories, "
        "Gherkin, or any internal factory tooling. "
        "Do not include fenced code blocks for the overall document — "
        "only use fenced blocks for code examples inside the README."
    )
    prompt = f"""You are writing a README.md for a software product.
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
    readme_content = call_llm(prompt, system=system, model=README_MODEL or None)
    README_PATH.write_text(readme_content)
    log(f"README.md written to {README_PATH}.")
