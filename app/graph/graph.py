from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.graph.state import AgentState

from app.graph.nodes import (
    router_node,
    validation_node,
    resolve_master_links_node,
    approval_node,
    execute_tool_node,
    responder_node,
    error_node,
    route_after_router,
    route_after_validation,
    route_after_master_link_resolution,
    route_after_approval,
)


def build_graph():

    builder = StateGraph(AgentState)

    # ------------------------------------
    # Nodes
    # ------------------------------------

    builder.add_node(
        "router",
        router_node,
    )

    builder.add_node(
        "validate",
        validation_node,
    )

    builder.add_node(
        "resolve_master_links",
        resolve_master_links_node,
    )

    builder.add_node(
        "approval",
        approval_node,
    )

    builder.add_node(
        "execute",
        execute_tool_node,
    )

    builder.add_node(
        "responder",
        responder_node,
    )

    builder.add_node(
        "error",
        error_node,
    )

    # ------------------------------------
    # START
    # ------------------------------------

    builder.add_edge(
        START,
        "router",
    )

    # ------------------------------------
    # Router
    # ------------------------------------

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "responder": "responder",
            "validate": "validate",
        },
    )

    # ------------------------------------
    # Validation
    # ------------------------------------

    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "error": "error",
            "approval": "resolve_master_links",
        },
    )

    builder.add_conditional_edges(
        "resolve_master_links",
        route_after_master_link_resolution,
        {
            "error": "error",
            "approval": "approval",
        },
    )

    # ------------------------------------
    # Approval
    # ------------------------------------

    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "error": "error",
            "execute": "execute",
        },
    )

    # ------------------------------------
    # Tool
    # ------------------------------------

    builder.add_edge(
        "execute",
        "responder",
    )

    # ------------------------------------
    # End
    # ------------------------------------

    builder.add_edge(
        "responder",
        END,
    )

    builder.add_edge(
        "error",
        END,
    )

    return builder.compile()


graph = build_graph()
