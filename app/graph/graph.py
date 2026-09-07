from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    conversation_context_node,
    coordinator_node,
    record_turn_node,
    route_after_coordinator,
    sales_mcp_node,
    unavailable_domain_node,
)
from app.graph.state import AgentState


def build_graph():
    """Route Sales requests to the controlled Sales MCP integration."""
    builder = StateGraph(AgentState)
    builder.add_node("conversation_context", conversation_context_node)
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("sales_mcp", sales_mcp_node)
    builder.add_node("unavailable_domain", unavailable_domain_node)
    builder.add_node("record_turn", record_turn_node)
    builder.add_edge(START, "conversation_context")
    builder.add_edge("conversation_context", "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "sales_mcp": "sales_mcp",
            "unavailable_domain": "unavailable_domain",
            "record_turn": "record_turn",
        },
    )
    builder.add_edge("sales_mcp", "record_turn")
    builder.add_edge("unavailable_domain", "record_turn")
    builder.add_edge("record_turn", END)
    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
