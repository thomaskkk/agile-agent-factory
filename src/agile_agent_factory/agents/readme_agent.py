from agile_agent_factory.config import (
    PRODUCT_ROOT, BP_BUSINESS_INTENT, BP_ARCH_DECISIONS, BP_ARCH_CONSTRAINTS,
    README_MAX_FILES, README_MAX_FILE_CHARS, README_MAX_TOTAL_CHARS,
)
from agile_agent_factory.agents.contract import AgentResult
from agile_agent_factory.tools.llm_adapters.readme import generate as generate_readme_text
from agile_agent_factory.tools.logger import log

BUSINESS_IDEA_PATH = PRODUCT_ROOT / "business_idea.md"
README_PATH = PRODUCT_ROOT / "README.md"
PYPROJECT_PATH = PRODUCT_ROOT / "pyproject.toml"


def generate_readme(state: dict) -> AgentResult:
    log("Generating README.md for product output.")

    blueprint_parts = [
        p.read_text() for p in (BP_BUSINESS_INTENT, BP_ARCH_DECISIONS, BP_ARCH_CONSTRAINTS)
        if p.exists()
    ]
    blueprint = "\n\n".join(blueprint_parts)
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
        f"### {path}\n```python\n{content[:README_MAX_FILE_CHARS]}\n```"
        for path, content in list(file_contents.items())[:README_MAX_FILES]
    )[:README_MAX_TOTAL_CHARS]

    readme_content = generate_readme_text(business_idea, blueprint, pyproject, files_block)
    README_PATH.write_text(readme_content)
    log(f"README.md written to {README_PATH}.")
    return AgentResult(success=True)
