import importlib


def test_wip_limits_honor_env_override(monkeypatch):
    monkeypatch.setenv("WIP_LIMIT_DEVELOPMENT", "7")
    import agile_agent_factory.config as config
    importlib.reload(config)
    assert config.WIP_LIMITS["development"] == 7
    # restore module state for other tests
    monkeypatch.delenv("WIP_LIMIT_DEVELOPMENT")
    importlib.reload(config)


def test_wip_limits_default_values():
    import agile_agent_factory.config as config
    assert config.WIP_LIMITS == {
        "refinement": 3,
        "tech_design": 2,
        "development": 2,
        "testing": 2,
        "code_review": 1,
    }
