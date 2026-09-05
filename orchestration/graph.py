"""
SatQueryAI - LangGraph orchestration skeleton.

    START -> boss -> (route_from_boss) -+-> worker1 -+
                                        +-> worker2 -+-> finalize -> END
                                        +-> worker3 -+
                                        +-> reject  -+

THE BOSS decides, router.route_from_boss dispatches, one specialist runs (or
none), finalize assembles the response and the auditable execution trace.

Worker nodes are stubs owned by nodes/worker_node.py. Teammates replace them by
passing their own callables to build_graph(worker_impls=...) - no edit to this
file, no edit to the router, no edit to the contract.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from orchestration.nodes.boss_node import boss_node
from orchestration.nodes.terminal_nodes import finalize_node, reject_node
from orchestration.nodes.worker_node import WorkerNode, default_worker_stubs
from orchestration.router import (
    NODE_BOSS,
    NODE_FINALIZE,
    NODE_REJECT,
    ROUTE_TARGETS,
    WORKER_NODE_NAMES,
    route_from_boss,
)
from orchestration.state import GraphState, InputBundle, WorkerId, initial_state


def build_graph(
    worker_impls: Optional[Dict[WorkerId, WorkerNode]] = None,
    boss_impl: Optional[Callable[[GraphState], Dict[str, Any]]] = None,
    checkpointer: Any = None,
):
    """
    Assemble and compile the graph.

    Args:
        worker_impls: WorkerId -> node callable. Missing entries fall back to the
            stub, so a half-finished team still has a runnable pipeline.
        boss_impl: overrides the Qwen-backed boss node. Use
            boss_node.make_boss_node(fn) to drive the graph from a scripted
            router in tests, which avoids loading 3B weights.
        checkpointer: optional LangGraph checkpointer for the streaming API.
    """
    stubs = default_worker_stubs()
    impls: Dict[WorkerId, WorkerNode] = {**stubs, **(worker_impls or {})}

    graph = StateGraph(GraphState)
    graph.add_node(NODE_BOSS, boss_impl or boss_node)
    for worker_id, node_name in WORKER_NODE_NAMES.items():
        graph.add_node(node_name, impls[worker_id])
    graph.add_node(NODE_REJECT, reject_node)
    graph.add_node(NODE_FINALIZE, finalize_node)

    graph.add_edge(START, NODE_BOSS)
    graph.add_conditional_edges(NODE_BOSS, route_from_boss, ROUTE_TARGETS)
    for node_name in WORKER_NODE_NAMES.values():
        graph.add_edge(node_name, NODE_FINALIZE)
    graph.add_edge(NODE_REJECT, NODE_FINALIZE)
    graph.add_edge(NODE_FINALIZE, END)

    return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()


def run_query(
    compiled_graph,
    query: str,
    inputs: InputBundle,
    request_id: str = "",
) -> Dict[str, Any]:
    """Convenience wrapper for backend/routes/query.py."""
    final_state = compiled_graph.invoke(initial_state(query, inputs, request_id))
    return final_state.get("final_response") or {}


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    import json

    from orchestration.state import ImageFormat, ImageMeta, Modality

    bundle = InputBundle(
        images=[
            ImageMeta(
                image_id="img_0",
                path="/data/img_0.tif",
                modality=Modality.OPTICAL,
                format=ImageFormat.GEOTIFF,
                georeferenced=True,
            )
        ]
    )
    app = build_graph()
    print(json.dumps(run_query(app, "Describe this scene.", bundle), indent=2))
