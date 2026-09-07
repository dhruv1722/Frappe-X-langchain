from typing import Annotated, TypedDict

from app.memory import ConversationTurn, append_conversation_turns


class AgentState(TypedDict, total=False):
    user_message: str
    conversation_context: str
    conversation_history: Annotated[
        list[ConversationTurn], append_conversation_turns
    ]
    pending_approval: dict[str, str] | None
    prepared_approval: dict[str, str | int] | None
    clear_pending_approval: bool

    today: str
    route: str
    selected_agent: str

    tool_name: str
    tool_parameters: dict

    tool_executed: bool
    tool_result: dict
    presentation: dict

    validation_error: str | None

    answer: str

    requires_approval: bool
    approved: bool
