import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent.parent   # agile_agent_factory/ → src/ → repo root
PRODUCT_ROOT = _REPO_ROOT.parent                    # repo root → Factory_project_claude/
CHECKPOINT_DB = _REPO_ROOT / "pipeline_checkpoint.db"

BLUEPRINT_DIR        = _REPO_ROOT / "blueprint"
BP_CONTEXT_DIR       = BLUEPRINT_DIR / "context"
BP_ARCHITECTURE_DIR  = BLUEPRINT_DIR / "architecture"
BP_TASKS_DIR         = BLUEPRINT_DIR / "tasks"
BP_QA_CRITERIA_DIR   = BP_CONTEXT_DIR / "qa_criteria"

BP_BUSINESS_INTENT   = BP_CONTEXT_DIR / "business_intent.md"
BP_UX_DECISIONS      = BP_CONTEXT_DIR / "ux_decisions.md"
BP_ARCH_DECISIONS    = BP_ARCHITECTURE_DIR / "decisions.md"
BP_ARCH_CONSTRAINTS  = BP_ARCHITECTURE_DIR / "constraints.md"


def bp_qa_criteria_path(story_key: str) -> Path:
    return BP_QA_CRITERIA_DIR / f"{story_key}.md"


def bp_task_path(story_key: str) -> Path:
    return BP_TASKS_DIR / f"{story_key}.md"

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "")
JIRA_USER_EMAIL = os.environ.get("JIRA_USER_EMAIL", "")
JIRA_API_KEY = os.environ.get("JIRA_API_KEY", "")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "")
JIRA_HUMAN_ACCOUNT_ID = os.environ.get("JIRA_HUMAN_ACCOUNT_ID", "")
JIRA_FLAG_FIELD_ID = os.environ.get("JIRA_FLAG_FIELD_ID", "customfield_10021")
JIRA_FLAG_VALUE = os.environ.get("JIRA_FLAG_VALUE", "Impediment")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
LLM_PRIMARY_PROVIDER = os.environ.get("LLM_PRIMARY_PROVIDER", "anthropic").lower()
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16384"))
# When both providers hit a quota/rate-limit (often transient per-minute limits,
# e.g. after aider saturates the shared Anthropic pool), back off and retry the
# whole provider chain this many times before pausing the pipeline.
LLM_QUOTA_MAX_RETRIES = int(os.environ.get("LLM_QUOTA_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF_SECONDS = int(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", "30"))

AIDER_ENABLED = os.environ.get("AIDER_ENABLED", "false").lower() == "true"
AIDER_MODEL = os.environ.get("AIDER_MODEL", "anthropic/claude-sonnet-4-6")

# Per-agent model overrides. Provider auto-detected from name:
#   claude-* → anthropic,  gpt-*/o1-*/o3-*/o4-* → openai
# Leave blank ("") to fall through to the global ANTHROPIC_MODEL / OPENAI_MODEL.
PO_MODEL = os.environ.get("PO_MODEL", "")
QA_MODEL = os.environ.get("QA_MODEL", "")
UX_MODEL = os.environ.get("UX_MODEL", "")
TL_MODEL = os.environ.get("TL_MODEL", "")
DEV_MODEL = os.environ.get("DEV_MODEL", "")
TEST_MODEL = os.environ.get("TEST_MODEL", "")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "")
README_MODEL = os.environ.get("README_MODEL", "")

MAX_RETRIES_DEV = int(os.environ.get("MAX_RETRIES_DEV", "2"))
MAX_REVIEW_RETRIES = int(os.environ.get("MAX_REVIEW_RETRIES", "2"))
MAX_CORRECTION_FAILURES = int(os.environ.get("MAX_CORRECTION_FAILURES", "2"))
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"
UV_BIN = os.environ.get("UV_BIN", "uv")

# Kanban WIP limits (Phase 3+)
WIP_LIMIT_REFINEMENT = int(os.environ.get("WIP_LIMIT_REFINEMENT", "3"))
WIP_LIMIT_TECH_DESIGN = int(os.environ.get("WIP_LIMIT_TECH_DESIGN", "2"))
WIP_LIMIT_DEVELOPMENT = int(os.environ.get("WIP_LIMIT_DEVELOPMENT", "2"))
WIP_LIMIT_TESTING = int(os.environ.get("WIP_LIMIT_TESTING", "2"))
WIP_LIMIT_CODE_REVIEW = int(os.environ.get("WIP_LIMIT_CODE_REVIEW", "1"))
