from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_jira():
    """A MagicMock JiraClient with DRY_RUN-safe no-op writes."""
    jira = MagicMock()
    jira.dry_run = True
    return jira


@pytest.fixture
def story_state():
    """Builder for a per-story StoryState dict with sensible defaults."""
    def _build(**overrides):
        base = {
            "story_key": "PROJ-1",
            "epic_key": "PROJ-0",
            "column": "refinement",
            "has_ui": False,
        }
        base.update(overrides)
        return base
    return _build
