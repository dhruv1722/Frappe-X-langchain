"""MCP-only nodes exposed to the LangGraph workflow.

ERP data access belongs to domain MCP servers. This module deliberately has no
Frappe REST client, static tool registry, or business-operation implementation.
"""

from app.agents.coordinator.agent import coordinator_node
from app.agents.sales.mcp_agent import sales_mcp_node
from app.graph.state import AgentState
from app.memory import build_conversation_context


def conversation_context_node(state: AgentState) -> AgentState:
    """Load the previous bounded transcript from the LangGraph checkpoint."""
    return {
        **state,
        "tool_name": "",
        "tool_parameters": {},
        "tool_executed": False,
        "tool_result": {},
        "presentation": {},
        "answer": "",
        "conversation_context": build_conversation_context(
            state.get("conversation_history")
        ),
    }


def route_after_coordinator(state: AgentState) -> str:
    """Send each implemented domain to its own MCP-backed specialist."""
    selected_agent = state.get("selected_agent")
    if selected_agent == "sales":
        return "sales_mcp"
    if selected_agent in {"purchase", "stock", "accounts"}:
        return "unavailable_domain"
    return "record_turn"


def unavailable_domain_node(state: AgentState) -> AgentState:
    """Do not fall back to REST while a domain MCP server is absent."""
    domain = str(state.get("selected_agent", "requested")).title()
    return {
        **state,
        "answer": (
            f"The {domain} MCP specialist is not configured yet, so I cannot "
            "access ERPNext for that request."
        ),
        "tool_executed": False,
    }


def record_turn_node(state: AgentState) -> AgentState:
    """Persist only the completed user/assistant turn in the checkpoint."""
    pending_approval = state.get("pending_approval")
    if state.get("clear_pending_approval"):
        pending_approval = None
    elif state.get("prepared_approval"):
        pending_approval = state["prepared_approval"]

    answer = state.get("answer", "").strip()
    history = [{"role": "user", "content": state["user_message"]}]
    if answer:
        history.append({"role": "assistant", "content": answer})
    return {
        **state,
        "conversation_history": history,
        "pending_approval": pending_approval,
        "prepared_approval": None,
        "clear_pending_approval": False,
    }
