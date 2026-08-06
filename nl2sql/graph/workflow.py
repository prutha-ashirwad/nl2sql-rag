"""LangGraph workflow definition.

analyze ─┬─▶ retrieve ─┬─▶ generate ─┬─▶ validate ─┬─▶ execute ─▶ finalize
         │             │             │             │
         │             │             │             └─▶ repair ──┘
         └─────────────┴─────────────┴────────────────▶ finalize
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nl2sql.config import Settings
from nl2sql.graph.nodes import (
    NODE_ANALYZE,
    NODE_EXECUTE,
    NODE_FINALIZE,
    NODE_GENERATE,
    NODE_REPAIR,
    NODE_RETRIEVE,
    NODE_VALIDATE,
    NL2SQLNodes,
    build_validation_router,
    route_after_analysis,
    route_after_generation,
    route_after_retrieval,
)
from nl2sql.graph.state import NL2SQLState
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)


def build_workflow(
    nodes: NL2SQLNodes, settings: Settings, *, execute: bool
) -> StateGraph:
    """Assemble and compile the graph, running validated queries when ``execute``."""
    graph: StateGraph = StateGraph(NL2SQLState)

    graph.add_node(NODE_ANALYZE, nodes.analyze_question)
    graph.add_node(NODE_RETRIEVE, nodes.retrieve_context)
    graph.add_node(NODE_GENERATE, nodes.generate_sql)
    graph.add_node(NODE_VALIDATE, nodes.validate_sql)
    graph.add_node(NODE_REPAIR, nodes.repair_sql)
    graph.add_node(NODE_EXECUTE, nodes.execute_sql)
    graph.add_node(NODE_FINALIZE, nodes.finalize)

    graph.add_edge(START, NODE_ANALYZE)

    graph.add_conditional_edges(
        NODE_ANALYZE,
        route_after_analysis,
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_FINALIZE: NODE_FINALIZE},
    )
    graph.add_conditional_edges(
        NODE_RETRIEVE,
        route_after_retrieval,
        {NODE_GENERATE: NODE_GENERATE, NODE_FINALIZE: NODE_FINALIZE},
    )
    graph.add_conditional_edges(
        NODE_GENERATE,
        route_after_generation,
        {NODE_VALIDATE: NODE_VALIDATE, NODE_FINALIZE: NODE_FINALIZE},
    )
    graph.add_conditional_edges(
        NODE_VALIDATE,
        build_validation_router(settings, execute),
        {
            NODE_EXECUTE: NODE_EXECUTE,
            NODE_REPAIR: NODE_REPAIR,
            NODE_FINALIZE: NODE_FINALIZE,
        },
    )

    # This cycle terminates because the router above stops routing to repair.
    graph.add_edge(NODE_REPAIR, NODE_VALIDATE)
    graph.add_edge(NODE_EXECUTE, NODE_FINALIZE)
    graph.add_edge(NODE_FINALIZE, END)

    compiled = graph.compile()
    logger.debug("NL2SQL workflow compiled (execution %s)", "on" if execute else "off")
    return compiled
