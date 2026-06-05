from agile_agent_factory.tools.llm_adapters.qa import build_qa_prompt
from agile_agent_factory.tools.llm_adapters.po import build_po_prompt
from agile_agent_factory.tools.llm_adapters.ux import build_ux_prompt
from agile_agent_factory.tools.llm_adapters.tl import build_tl_prompt
from agile_agent_factory.tools.llm_adapters.reviewer import build_review_prompt
from agile_agent_factory.tools.llm_adapters.readme import build_readme_prompt


def test_qa_prompt_includes_summary_and_description():
    prompt = build_qa_prompt("Login form", "User can log in")
    assert "Login form" in prompt
    assert "User can log in" in prompt


def test_qa_prompt_omits_empty_description():
    prompt = build_qa_prompt("Login form")
    assert "Story description for additional context" not in prompt


def test_po_prompt_includes_idea_and_feedback():
    prompt = build_po_prompt("Build a todo app", hitl_feedback="Linux only")
    assert "Build a todo app" in prompt
    assert "Linux only" in prompt


def test_po_prompt_omits_empty_feedback():
    prompt = build_po_prompt("Build a todo app")
    assert "Human clarification" not in prompt


def test_ux_prompt_includes_idea_and_stories():
    prompt = build_ux_prompt("A recipe manager", "**PROJ-1:** List recipes")
    assert "A recipe manager" in prompt
    assert "List recipes" in prompt


def test_tl_prompt_includes_story_keys_and_ready_contracts():
    prompt = build_tl_prompt(
        "Idea text", ["PROJ-1"], ux_spec={}, ready_contracts={"PROJ-1": {"x": 1}}
    )
    assert "PROJ-1" in prompt
    assert "Idea text" in prompt


def test_tl_prompt_includes_ux_block_when_ui_present():
    prompt = build_tl_prompt(
        "Idea", ["PROJ-1"], ux_spec={"ui_type": "web", "technology": "Flask"}, ready_contracts={}
    )
    assert "UI/UX Design Specification" in prompt


def test_review_prompt_includes_dod_and_files():
    prompt = build_review_prompt("DoD: must work", "### app/x.py", write_scope=None)
    assert "DoD: must work" in prompt
    assert "app/x.py" in prompt


def test_review_prompt_includes_write_scope_when_provided():
    prompt = build_review_prompt("DoD", "files", write_scope=["app/owned.py"])
    assert "app/owned.py" in prompt
    assert "Story write scope" in prompt


def test_readme_prompt_includes_business_idea_and_setup_rule():
    prompt = build_readme_prompt("Idea", "blueprint", "[project]", "### app/main.py")
    assert "Idea" in prompt
    assert "## Setup" in prompt
