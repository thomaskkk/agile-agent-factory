# Agile-Agent-Factory

An autonomous "Dark Factory" software development pipeline. Given a business idea in plain English, it orchestrates end-to-end engineering: requirement refinement, Jira backlog creation, code generation, test-driven iteration, code review, and emulated CI/CD deployment — with safe Human-in-the-Loop interruption at every phase boundary.

## How it works

```
business_idea.md
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  init → po                                                          │
│  (reads business_idea.md, provisions Jira Epics/Stories)            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  dispatcher (kanban loop — runs after every agent, right-to-left)   │
│                                                                     │
│  Backlog → Refinement → Tech Design → Development → Testing         │
│              QA + UX        TL           Dev          Test          │
│            (parallel)    (batch)                    → Code Review   │
│                                                         Review      │
│                                                       → Done        │
│                                                                     │
│  WIP limits enforced per column; Send() fan-out for parallelism     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ (all stories done)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  finalize                                                           │
│  README generation + Emulated SRE CI/CD → Jira Done                │
└─────────────────────────────────────────────────────────────────────┘
```

**HITL pause states:** If ambiguity is detected (PO phase), or downstream pytest retries are exhausted, the pipeline pauses via LangGraph `interrupt()` and sets a Jira flag with an `@mention`. Re-running `main.py` issues `Command(resume=feedback)` to resume from the interrupted node after the flag is cleared. If either LLM provider hits a quota/rate-limit error the pipeline pauses with the same interrupt/resume mechanism.

## Directory layout

```text
agile-agent-factory/        ← repository root
├── business_idea.md        ← tracked example input
├── app/                    ← generated production code (gitignored)
├── tests/                  ← generated product tests (gitignored)
└── factory/                ← tracked factory implementation
    ├── main.py             ← entry point: LangGraph compile/resume, HITL Command(resume=) dispatch
    ├── pyproject.toml      ← hatchling src layout; pytest pythonpath = ["src"]
    ├── blueprint/          ← layered handoff blueprint (generated; gitignored)
    │   ├── context/        ← business_intent.md, ux_decisions.md, qa_criteria/<story>.md
    │   ├── architecture/   ← decisions.md, constraints.md
    │   └── tasks/          ← per-story task files (<story_key>.md)
    ├── src/
    │   └── agile_agent_factory/
    │       ├── config.py       ← env loader from factory/.env + repo-root PRODUCT_ROOT
    │       ├── state.py        ← PipelineState + StoryState TypedDicts; merge_stories reducer
    │       ├── graph.py        ← StateGraph definition (init→po→dispatcher→agents→finalize)
    │       ├── tools/          ← infrastructure utilities
    │       ├── agents/         ← business-logic agents (each returns AgentResult)
    │       └── nodes/          ← LangGraph node wrappers
    └── tests/              ← factory unit tests (tracked)
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Jira Cloud project with API token
- Anthropic API key (OpenAI optional, used as fallback)
- [Aider](https://aider.chat/) (optional — `pip install aider-chat`; required only when `AIDER_ENABLED=true`)

## Setup

```bash
cd agile-agent-factory/factory
uv sync

cp .env.example .env
# Edit .env — fill in JIRA_* and ANTHROPIC_API_KEY at minimum
```

## Running

```bash
# First run (DRY_RUN=true in .env — safe, no real Jira writes)
uv run main.py

# Reset state without touching generated code
uv run main.py --reset-state

# Full live run (set DRY_RUN=false in .env first)
uv run main.py --reset-state && uv run main.py
```

## Configuration

All settings live in `.env`. Key variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DRY_RUN` | `true` | When true, all Jira writes are logged but not executed |
| `ANTHROPIC_API_KEY` | — | Required for LLM calls |
| `JIRA_BASE_URL` | — | e.g. `https://yourcompany.atlassian.net` |
| `JIRA_USER_EMAIL` | — | Jira account email |
| `JIRA_API_KEY` | — | Jira API token |
| `JIRA_PROJECT_KEY` | — | e.g. `PROJ` |
| `JIRA_HUMAN_ACCOUNT_ID` | — | Account ID to `@mention` on HITL blocks |
| `AIDER_ENABLED` | `false` | Enable Aider CLI for code generation (requires `aider` on PATH) |
| `AIDER_MODEL` | `anthropic/claude-sonnet-4-6` | Model passed to Aider via `--model` |
| `UV_BIN` | `uv` | Full path to `uv` binary (e.g. `/snap/bin/uv`) |
| `MAX_RETRIES_DEV` | `2` | pytest failure retries before HITL pause |
| `MAX_REVIEW_RETRIES` | `2` | code-review rejection retries per story before HITL pause |
| `MAX_CORRECTION_FAILURES` | `2` | correction-loop zero-file failures before HITL (does not consume `MAX_RETRIES_DEV` budget) |
| `MAX_STRATEGY_RETRIES` | `3` | separate budget for mechanical recovery (dep re-resolve, stub scaffold, truncation retry); never consumes `MAX_RETRIES_DEV` |
| `ASSUMPTION_RISK_THRESHOLD` | `0.7` | PO risk gate: aggregated assumption confidence above this escalates to HITL; below → proceed and post ledger to Jira |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model ID |
| `OPENAI_MODEL` | `gpt-5.4` | OpenAI model ID (fallback) |
| `OPENAI_API_KEY` | — | Required only if Anthropic calls fail |
| `LLM_PRIMARY_PROVIDER` | `anthropic` | Primary LLM provider (`anthropic` or `openai`) |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-call LLM timeout; raise if you see repeated timeouts on large codegen |
| `LLM_MAX_TOKENS` | `16384` | Max output tokens per LLM call; raise if multi-file responses are truncated |
| `LLM_QUOTA_MAX_RETRIES` | `3` | How many times to retry the full provider chain on quota/rate-limit before pausing |
| `LLM_RETRY_BACKOFF_SECONDS` | `30` | Base back-off; actual wait is `base × 2^attempt` |
| `WIP_LIMIT_REFINEMENT` | `3` | Max stories in the refinement column simultaneously |
| `WIP_LIMIT_TECH_DESIGN` | `2` | Max stories in the tech design column simultaneously |
| `WIP_LIMIT_DEVELOPMENT` | `2` | Max stories in the development column simultaneously |
| `WIP_LIMIT_TESTING` | `2` | Max stories in the testing column simultaneously |
| `WIP_LIMIT_CODE_REVIEW` | `1` | Max stories in the code review column simultaneously |
| `PO_MODEL` / `TL_MODEL` / `DEV_MODEL` / `TEST_MODEL` / `REVIEWER_MODEL` | `claude-sonnet-4-6` | Per-agent model override (quality-critical steps). Provider auto-detected: `claude-*` → Anthropic, `gpt-*`/`o*-*` → OpenAI |
| `QA_MODEL` / `UX_MODEL` / `README_MODEL` | `claude-haiku-4-5-20251001` | Per-agent model override (structured/low-risk steps). Blank falls through to `ANTHROPIC_MODEL` / `OPENAI_MODEL` |
| `LANGCHAIN_TRACING_V2` | `true` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key (optional) |
| `LANGCHAIN_PROJECT` | `agile-agent-factory` | LangSmith project name |

## State machine

The pipeline uses LangGraph `interrupt()` / `Command(resume=feedback)` for HITL:

```
(pipeline running)
       │
       ├─▶ PO detects ambiguity ──────────────────────▶ interrupt("refinement")
       │                                                  Jira flag + @mention
       │                                                  re-run to resume
       │
       ├─▶ pytest retries exhausted ─────────────────▶ interrupt("intervention")
       │                                                  Jira flag + @mention
       │                                                  re-run to resume
       │
       ├─▶ code review retries exhausted ───────────▶ interrupt("intervention")
       │                                                  Jira flag + @mention
       │                                                  re-run to resume
       │
       └─▶ LLM quota exceeded ──────────────────────▶ interrupt("quota")
                                                        Jira flag + @mention
                                                        resolve quota, clear flag,
                                                        re-run to resume
```

State is persisted in `factory/pipeline_checkpoint.db` via LangGraph `SqliteSaver` (gitignored). `--reset-state` deletes the database; generated product code in repo-root `app/` and `tests/` is never touched by reset.

## Running the tests

```bash
cd agile-agent-factory/factory
uv run pytest tests/ -v
```

321 tests covering `config` (7), `workflow` (2), `jira_facade` (5), `llm_adapters` (10), `agent_contract` (2), `path_utils` (13), `llm_client` (27), `jira_client` (17), `pytest_runner` (7), `po_agent` (10), `tl_agent` (8), `qa_agent` (4), `ux_agent` (7), `reviewer_agent` (17), `failure_recovery` (35), `review_node` (25), `test_node` (13), `dev_node` (18), `dependencies` (6), `aider_client` (5), `readme_agent` (2), `ready_contract` (18), `dispatcher` (22), and `graph` (41).
