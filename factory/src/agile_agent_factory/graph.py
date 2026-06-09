"""LangGraph StateGraph definition for the agile-agent-factory pipeline.

Phase 1: sequential PO→QA→UX→TL→Dev→Test→Review→README→Deploy flow.
Phase 3: replaced by kanban dispatcher with right-to-left priority routing.
Phase 5: testing replaced by a dedicated retry subgraph.
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agile_agent_factory.config import MAX_REVIEW_RETRIES, CHECKPOINT_DB
from agile_agent_factory.nodes import (
    dev_node,
    dispatch_stories,
    finalize_node,
    init_node,
    po_node,
    qa_node,
    refinement_gate_node,
    review_node,
    test_node,
    tl_node,
    ux_node,
)
from agile_agent_factory.tools.logger import log
from agile_agent_factory.state import PipelineState


# ---------------------------------------------------------------------------
# Routing functions (deterministic — no LLM calls)
# ---------------------------------------------------------------------------

def _route_dispatcher(state: PipelineState):
    """Route from dispatcher node based on dispatch_stories() output.

    Returns either a list of Send() objects (fan-out to agent nodes) or
    the string "finalize" when all stories are done.
    """
    result = dispatch_stories(state)
    if result == "finalize":
        return "finalize"
    if isinstance(result, list) and result:
        return result  # list of Send() objects
    # Empty list means interrupt already fired (execution suspended before return)
    # or a WIP-limit deadlock (should not happen in Phase 3). Fall through to END.
    log("WARNING: dispatcher returned no sends and no finalize — ending graph.")
    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Compile and return the pipeline StateGraph.

    Args:
        checkpointer: override the default SqliteSaver (useful for tests that
                      pass InMemorySaver to avoid disk writes).
    """
    g = StateGraph(PipelineState)

    # Register nodes
    g.add_node("init", init_node)
    g.add_node("po", po_node)
    g.add_node("dispatcher", lambda state: {})  # no-op pass-through; routing done in edge
    g.add_node("qa", qa_node)
    g.add_node("ux", ux_node)
    g.add_node("refinement_gate", refinement_gate_node)
    g.add_node("tl", tl_node)
    g.add_node("dev", dev_node)
    g.add_node("test", test_node)
    g.add_node("review", review_node)
    g.add_node("finalize", finalize_node)

    # Entry: init → po → dispatcher
    g.add_edge(START, "init")
    g.add_edge("init", "po")
    g.add_edge("po", "dispatcher")

    # Dispatcher fans out to agents (or routes to finalize)
    g.add_conditional_edges(
        "dispatcher",
        _route_dispatcher,
        {
            "qa": "qa",
            "ux": "ux",
            "refinement_gate": "refinement_gate",
            "tl": "tl",
            "dev": "dev",
            "test": "test",
            "review": "review",
            "finalize": "finalize",
            END: END,
        },
    )

    # All agent nodes loop back to the dispatcher
    g.add_edge("qa", "dispatcher")
    g.add_edge("ux", "dispatcher")
    g.add_edge("refinement_gate", "dispatcher")
    g.add_edge("tl", "dispatcher")
    g.add_edge("dev", "dispatcher")
    g.add_edge("test", "dispatcher")
    g.add_edge("review", "dispatcher")

    # Pipeline terminates after finalize
    g.add_edge("finalize", END)

    from langgraph.checkpoint.base import BaseCheckpointSaver
    if not isinstance(checkpointer, (BaseCheckpointSaver, bool)) and checkpointer is not None:
        checkpointer = None  # Studio passes a config dict; fall through to SQLite

    if checkpointer is None:
        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)

