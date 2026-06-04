# Agile-Agent-Factory

An autonomous "Dark Factory" software development pipeline. Given a business idea in plain English, it orchestrates end-to-end engineering: requirement refinement, Jira backlog creation, code generation, test-driven iteration, code review, and emulated CI/CD deployment — with safe Human-in-the-Loop interruption at every phase boundary.

## How it works

```
../business_idea.md
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

```
agile-agent-factory/        ← this repo (the factory)
├── main.py                 ← entry point: LangGraph compile/resume, HITL Command(resume=) dispatch
├── pyproject.toml          ← hatchling src layout; pytest pythonpath = ["src"]
├── blueprint/              ← layered handoff blueprint (generated; gitignored)
│   ├── context/            ← business_intent.md, ux_decisions.md, qa_criteria/<story>.md
│   ├── architecture/       ← decisions.md, constraints.md
│   └── tasks/              ← per-story task files (<story_key>.md)
├── src/
│   └── agile_agent_factory/
│       ├── config.py       ← env var loader + PRODUCT_ROOT, CHECKPOINT_DB, BLUEPRINT_PATH
│       ├── state.py        ← PipelineState + StoryState TypedDicts; merge_stories reducer
│       ├── graph.py        ← StateGraph definition (init→po→dispatcher→agents→finalize)
│       ├── tools/          ← infrastructure utilities
│       │   ├── logger.py           ← timestamped stdout logger
│       │   ├── llm_client.py       ← Anthropic primary + OpenAI fallback; LLMQuotaExceeded exception
│       │   ├── jira_client.py      ← Jira REST API v3 (ADF, JQL, transitions, flags, subtask type discovery)
│       │   ├── path_utils.py       ← safe path normalization for AI-generated paths
│       │   ├── pytest_runner.py    ← pytest subprocess with PYTHONPATH injection
│       │   ├── dependencies.py     ← third-party dependency resolution (declared + UX + AST scan)
│       │   └── aider_client.py     ← Aider CLI subprocess (gated by AIDER_ENABLED)
│       ├── agents/         ← business-logic agents
│       │   ├── po_agent.py         ← Jira epic/story provisioning, has_ui detection, upstream HITL
│       │   ├── qa_agent.py         ← Gherkin acceptance criteria per story
│       │   ├── ux_agent.py         ← UI/UX design spec (cli/web/tui), appends to Jira stories
│       │   ├── tl_agent.py         ← architecture design, dependency collection, subtasks, blueprint/ files
│       │   ├── reviewer_agent.py   ← DoD audit via LLM
│       │   ├── readme_agent.py     ← README generation from blueprint + product code scan
│       │   └── sre_agent.py        ← emulated CI/CD report, Jira Done transition
│       └── nodes/          ← LangGraph node wrappers
│           ├── pipeline.py         ← all node functions (per-story via active_story_key; TL is batch)
│           ├── dispatcher.py       ← right-to-left kanban dispatcher with WIP limits and Send() fan-out
│           └── subgraphs.py        ← pytest retry subgraph
└── tests/                  ← factory unit tests (not product tests)

../                         ← product output (sibling directory)
├── business_idea.md        ← INPUT: plain-English product description
├── app/                    ← OUTPUT: generated production code
└── tests/                  ← OUTPUT: generated product tests
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Jira Cloud project with API token
- Anthropic API key (OpenAI optional, used as fallback)
- [Aider](https://aider.chat/) (optional — `pip install aider-chat`; required only when `AIDER_ENABLED=true`)

## Setup

```bash
cd agile-agent-factory
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
       └─▶ LLM quota exceeded ──────────────────────▶ interrupt("quota")
                                                        Jira flag + @mention
                                                        resolve quota, clear flag,
                                                        re-run to resume
```

State is persisted in `pipeline_checkpoint.db` via LangGraph `SqliteSaver` (gitignored). `--reset-state` deletes the database; Generated product code in `../app/` and `../tests/` is never touched by reset.

## Running the tests

```bash
cd agile-agent-factory
uv run pytest tests/ -v
```

133 tests covering `path_utils` (8), `llm_client` (23), `jira_client` (17), `pytest_runner` (5), `po_agent` (4), `tl_agent` (5), `qa_agent` (2), `ux_agent` (7), `reviewer_agent` (6), `dependencies` (6), `aider_client` (5), `readme_agent` (2), `dispatcher` (18), and `graph` (25).
