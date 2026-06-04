from unittest.mock import patch

import pytest


def test_generate_readme_writes_file(tmp_path, monkeypatch):
    """generate_readme calls the LLM and writes the result to ../README.md."""
    import agile_agent_factory.agents.readme_agent as readme_agent

    monkeypatch.setattr(readme_agent, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    monkeypatch.setattr(readme_agent, "PRODUCT_ROOT", tmp_path)
    monkeypatch.setattr(readme_agent, "README_PATH", tmp_path / "README.md")

    (tmp_path / "handoff_blueprint.md").write_text("# Blueprint\nProduct: test product")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "tracker.py").write_text("def track(): pass")

    fake_readme = "# Test Product\n\nA test product.\n"
    with patch("agile_agent_factory.agents.readme_agent.call_llm", return_value=fake_readme):
        readme_agent.generate_readme({})

    result = (tmp_path / "README.md").read_text()
    assert result == fake_readme


def test_generate_readme_quota_propagates(tmp_path, monkeypatch):
    """LLMQuotaExceeded raised inside generate_readme must not be swallowed."""
    from agile_agent_factory.tools.llm_client import LLMQuotaExceeded
    import agile_agent_factory.agents.readme_agent as readme_agent

    monkeypatch.setattr(readme_agent, "BLUEPRINT_PATH", tmp_path / "handoff_blueprint.md")
    monkeypatch.setattr(readme_agent, "PRODUCT_ROOT", tmp_path)
    monkeypatch.setattr(readme_agent, "README_PATH", tmp_path / "README.md")

    (tmp_path / "handoff_blueprint.md").write_text("# Blueprint")

    with patch("agile_agent_factory.agents.readme_agent.call_llm", side_effect=LLMQuotaExceeded("anthropic", "quota")):
        with pytest.raises(LLMQuotaExceeded):
            readme_agent.generate_readme({})
