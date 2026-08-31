from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_message: str
    conversation_context: str

    today: str
    route: str

    tool_name: str
    tool_parameters: dict

    tool_executed: bool
    tool_result: dict

    validation_error: str | None

    answer: str

    requires_approval: bool
    approved: bool