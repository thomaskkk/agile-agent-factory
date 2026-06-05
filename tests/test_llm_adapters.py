from agile_agent_factory.tools.llm_adapters.qa import build_qa_prompt


def test_qa_prompt_includes_summary_and_description():
    prompt = build_qa_prompt("Login form", "User can log in")
    assert "Login form" in prompt
    assert "User can log in" in prompt
