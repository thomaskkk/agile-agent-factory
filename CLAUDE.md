# CLAUDE.md — Agile-Agent-Factory

## What this repo is

This is the **factory** (orchestrator), not the product. It reads `../business_idea.md` and writes generated code to `../app/` and `../tests/`. Never confuse the two:

| Path | Role |
| --- | --- |
| `agile-agent-factory/` | Factory source — the code you edit here |
| `../app/` | Generated product code — do not edit manually |
| `../tests/` | Generated product tests — do not edit manually |
| `../business_idea.md` | Pipeline input — edit to change what gets built |

## Running

```bash
cd agile-agent-factory
uv run main.py                # start or resume pipeline
uv run main.py --reset-state  # delete checkpoint DB and start fresh
uv run pytest tests/ -v       # run factory unit tests (not product tests)
```

## Environment

- Python 3.12+, uv package manager
- All config in `.env` (copy from `.env.example`, never commit `.env`)
- `DRY_RUN=true` by default — safe to run without real Jira credentials
- `AIDER_ENABLED=false` — aider binary not installed; downstream falls back to LLM-direct code generation
- `UV_BIN=uv` — change to full snap path (e.g. `/snap/bin/uv`) if uv is snap-installed and breaks in subprocess calls
- `LLM_TIMEOUT_SECONDS=120` — default raised from 60s; large codegen calls need the headroom
- `LLM_MAX_TOKENS=16384` — max output tokens per call; raise if multi-file responses get truncated
- `LLM_QUOTA_MAX_RETRIES=3` / `LLM_RETRY_BACKOFF_SECONDS=30` — exponential backoff before pausing the pipeline on quota errors
- `PO_MODEL` / `QA_MODEL` / `UX_MODEL` / `TL_MODEL` / `DEV_MODEL` / `TEST_MODEL` / `REVIEWER_MODEL` / `README_MODEL` — per-agent model override. Provider auto-detected from name prefix (`claude-*` → Anthropic, `gpt-*`/`o*-*` → OpenAI). Leave blank to use the global `ANTHROPIC_MODEL` / `OPENAI_MODEL`.
- `WIP_LIMIT_REFINEMENT=3` / `WIP_LIMIT_TECH_DESIGN=2` / `WIP_LIMIT_DEVELOPMENT=2` / `WIP_LIMIT_TESTING=2` / `WIP_LIMIT_CODE_REVIEW=1` — kanban WIP limits per column

## Architecture

LangGraph `StateGraph` with a kanban dispatcher for story-level parallelism:

- **State**: `PipelineState` (TypedDict) in `src/agile_agent_factory/state.py` — persisted by `SqliteSaver` in `pipeline_checkpoint.db`. Per-story state lives in `PipelineState.stories[story_key]` (a `StoryState` dict merged by `merge_stories` reducer).
- **Graph**: `src/agile_agent_factory/graph.py` — `init → po → dispatcher → [qa | ux | refinement_gate | tl | dev | test | review] → dispatcher (loop) → finalize → END`
- **Dispatcher**: `src/agile_agent_factory/nodes/dispatcher.py` — deterministic routing. Scans columns right-to-left (`code_review → testing → development → tech_design → refinement`), respects WIP limits, emits `Send()` commands. TL is dispatched as a batch (one `Send` for all `tech_design` stories); all other agents are per-story.
- **Nodes**: `src/agile_agent_factory/nodes/pipeline.py` — each wraps an existing agent. Returns a partial dict; LangGraph merges via reducers. Never calls `update_state()`.
- **HITL**: `langgraph.types.interrupt()` suspends execution; `Command(resume=feedback)` resumes. Three interrupt types: `refinement` (PO ambiguity), `intervention` (pytest retry exhaustion), `quota` (LLM rate-limit).

### Kanban Column Flow

```
Backlog → Refinement → Tech Design → Development → Testing → Code Review → Done
                         ↑                                        |
                         └────── review rejected (≤2 retries) ───┘
```

| Column | Agent | Notes |
| --- | --- | --- |
| `refinement` | QA + UX (parallel) | QA sets `refinement_qa_done`; UX sets `refinement_ux_done`. Dispatcher sends to `refinement_gate` when both done (avoids column-field race). UX skipped when `has_ui=False`. `refinement_gate` builds and validates a ready contract (DoR check) before advancing to `tech_design`; validation errors keep the story in `refinement`. |
| `tech_design` | TL (batch) | Processes ALL tech_design stories in one call; advances them all to development. |
| `development` | Dev | LLM or aider code generation. |
| `testing` | Test | Iterative pytest retry loop with two budgets (`MAX_RETRIES_DEV`, `MAX_CORRECTION_FAILURES`). |
| `code_review` | Review | LLM DoD audit. Approve → done. Reject → sets `review_status=rework_needed`, dev reworks in-place (up to `MAX_REVIEW_RETRIES=2`). Exhaustion → HITL. |
| `done` | Finalize | README generation + SRE deployment (runs once when all stories done). |

### Crash Recovery

State is fully persisted in `pipeline_checkpoint.db` after every node. On restart, LangGraph resumes from the last completed node automatically.

## HITL (Human-in-the-Loop)

All HITL flows use `interrupt()` / `Command(resume=feedback)`:

- **Ambiguity** (`refinement`): PO detects ambiguity → flags Jira issue → `interrupt()`. Human adds clarifying comment. On `main.py` re-run, Jira flag is cleared, comment fetched, and `Command(resume=feedback)` re-runs PO with `hitl_feedback` injected into the prompt.
- **Dev intervention** (`intervention`): pytest retries exhausted → flags Jira story → `interrupt()`. Human fixes code or leaves feedback. On resume, pytest runs first; if already passing, LLM generation is skipped.
- **Quota** (`quota`): either LLM provider rate-limited → flags Jira issue → `interrupt()`. Resolve quota, clear flag, re-run `main.py`.

## Path safety

All AI-generated file paths go through `normalize_generated_path()` in `src/agile_agent_factory/tools/path_utils.py`. It:
- Rejects absolute paths
- Strips leading `../` traversal sequences
- Collapses `app/app/` and `tests/tests/` duplicates
- Rejects anything that doesn't resolve under `../app/` or `../tests/`

Never bypass this function when writing files from LLM output.

## Jira client rules

- All searches use `POST /rest/api/3/search/jql` (never GET search)
- All comments use ADF format (never plain text)
- Transitions are discovered dynamically — never hardcode transition IDs
- `transition_issue` checks `DRY_RUN` at the top before calling `get_transitions` — no HTTP GET fires in dry-run mode
- Every write method checks `DRY_RUN` and logs `[DRY_RUN]` prefix when skipped

## Testing

Factory tests live in `agile-agent-factory/tests/`. They cover (158 tests total):
- `path_utils` — 10 tests (path normalization and traversal rejection)
- `llm_client` — 23 tests (mocked Anthropic + OpenAI, JSON parsing, quota propagation, exponential backoff retry, network-error handling)
- `jira_client` — 17 tests (mocked HTTP via `responses` library, subtask type discovery, `append_adf_doc`, `update_issue_description`, DRY_RUN-early skip)
- `pytest_runner` — 5 tests (PYTHONPATH injection, exit codes, `--with` extra-package flags)
- `po_agent` — 4 tests (idempotency guard, issue creation, hitl_feedback injection)
- `tl_agent` — 8 tests (architecture caching on resume, subtask idempotency, fresh LLM call + persistence, write-scope guard)
- `qa_agent` — 4 tests
- `ux_agent` — 7 tests (validated spec, Jira description append, quota propagation, unknown ui_type/technology rejection)
- `reviewer_agent` — 9 tests
- `dependencies` — 6 tests (import scan, package aliases, unparseable file skip, 3-signal union, Flask recovery)
- `aider_client` — 5 tests (is_available guards, subprocess args, failure exit code)
- `readme_agent` — 2 tests
- `ready_contract` — 12 tests (contract validation, readiness repair, test_contract extraction)
- `dispatcher` — 18 tests (RTL priority, WIP limits, refinement sub-phases, gate routing, active_story_key, code_review rework routing)
- `graph` — 28 tests (graph compilation, node behavior, routing, state reducer, HITL scenarios, review HITL exhaustion, dev rework path)

PYTHONPATH for product tests is injected by `src/agile_agent_factory/tools/pytest_runner.py` (points to `../`), so `from app.module import ...` resolves correctly from the product root.

## Key invariants

- `pipeline_checkpoint.db` is the single source of truth — LangGraph's `SqliteSaver` manages it; never write state directly
- `blueprint/` is regenerated every upstream run — never commit it (it's gitignored)
- `review_retries` in StoryState caps rework cycles at `MAX_REVIEW_RETRIES = 2`. Rejection keeps the story in `code_review` with `review_status="rework_needed"`; dev reworks in-place. At exhaustion, HITL fires (no force-accept). After resume, `review_retries` resets to 0.
- A story's `column` only ever advances forward: `refinement → tech_design → development → testing → code_review → done`. `review_status` sub-state (`"pending_review"` / `"rework_needed"`) drives rework routing without backward column movement.
- `dev_node` skips all Jira transitions when `story.column == "code_review"` (rework path)
- `po_node` is idempotent: returns `{}` immediately if `state["stories"]` is already populated
- `analyze_and_provision` returns early if `state["story_keys"]` is populated — never bypass this
- `state["epic_keys"]` is a **list** — every node that touches epics must iterate the full list
- `jira_client.get_subtask_issue_type()` queries the project API — never hardcode `"Subtask"`
- Quota errors (`LLMQuotaExceeded`) propagate from `src/agile_agent_factory/tools/llm_client.py` through agents to nodes — do not catch them inside agents; nodes handle them with `interrupt()`
- `qa_node` and `ux_node` only set their `refinement_qa_done` / `refinement_ux_done` flags — they do NOT advance the column. The dispatcher sends to `refinement_gate` when both are True, which advances to `tech_design`. This avoids a race condition when both run in parallel.
- `tl_node` is batch: collects ALL stories in `tech_design` column and processes them together; dispatcher emits one `Send("tl", state)` not one per story
- Architecture is cached in `StoryState.architecture` after the TL run — `design_architecture` reuses it on resume (no extra LLM call); subtasks already in `state["subtasks"]` are skipped
- `dependencies.resolve_dependencies(state, product_root)` unions three signals: TL-declared deps, UX-technology package, and a static AST scan of generated code — pass its result to `run_pytest()` as `extra_packages`
- `MAX_CORRECTION_FAILURES = 2` caps how many times the correction LLM may produce zero usable files before escalating to HITL (these do not consume the `MAX_RETRIES_DEV` budget)
- `active_story_key` in `PipelineState` is set by the dispatcher in each `Send()` payload — per-story nodes call `_active_story(state)` to get their assigned story; TL ignores it and reads all `tech_design` stories directly
- `refinement_gate_node` builds a `ready_contract` dict (via `agents/ready_contract.py`) and validates it before advancing a story to `tech_design`; validation errors set `ready_validation_errors` and keep the story in `refinement`
- TL task files (`blueprint/tasks/<story_key>.md`) include a **write-scope section** derived from the story's `test_contract` — it lists the exact files the dev agent is allowed to create/overwrite; the reviewer enforces the same scope via `write_scope` passed to `review_patch()`
- `StoryState` new fields: `test_contract` (dict — expected test names and target interfaces, set by QA), `ready_contract` (dict — full DoR snapshot), `ready_validation_errors` (list[str]), `ready_validated` (bool)

## Local-only files

- `PRD.md` — product requirements document; gitignored, never commit it. Edit locally to evolve requirements.

## Known issues (not yet fixed)

- (none currently tracked)
