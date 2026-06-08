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


def test_review_and_readme_budgets_present():
    import agile_agent_factory.config as config
    assert config.REVIEW_MAX_FILES == 30
    assert config.REVIEW_MAX_FILE_CHARS == 8000
    assert config.REVIEW_MAX_TOTAL_CHARS == 50000
    assert config.README_MAX_FILES == 15
    assert config.README_MAX_FILE_CHARS == 2000
    assert config.README_MAX_TOTAL_CHARS == 20000


def test_max_strategy_retries_default():
    import agile_agent_factory.config as config
    assert config.MAX_STRATEGY_RETRIES == 3


def test_max_strategy_retries_env_override(monkeypatch):
    import importlib
    monkeypatch.setenv("MAX_STRATEGY_RETRIES", "5")
    import agile_agent_factory.config as config
    importlib.reload(config)
    assert config.MAX_STRATEGY_RETRIES == 5
    monkeypatch.delenv("MAX_STRATEGY_RETRIES")
    importlib.reload(config)


def test_assumption_risk_threshold_default():
    import agile_agent_factory.config as config
    assert config.ASSUMPTION_RISK_THRESHOLD == 0.7


def test_max_refinement_retries_default():
    import agile_agent_factory.config as config
    assert config.MAX_REFINEMENT_RETRIES == 3


def test_max_refinement_retries_env_override(monkeypatch):
    import importlib
    monkeypatch.setenv("MAX_REFINEMENT_RETRIES", "5")
    import agile_agent_factory.config as config
    importlib.reload(config)
    assert config.MAX_REFINEMENT_RETRIES == 5
    monkeypatch.delenv("MAX_REFINEMENT_RETRIES")
    importlib.reload(config)


def test_model_chain_single_value_is_single_string():
    """Single-value model names (no comma) remain usable as single strings."""
    import agile_agent_factory.config as config
    # DEV_MODEL is a plain string — callers pass it as model=DEV_MODEL or None
    assert "," not in config.DEV_MODEL  # default is ""
