import json
from datetime import date

from app.llm import get_llm
from app.tools.frappe import (
    MasterLinkResolutionError,
    frappe_create_document,
    frappe_get_list,
    frappe_get_document,
    frappe_update_document,
    frappe_delete_document,
    frappe_get_doctype_definition,
    resolve_document_master_links,
)

from app.graph.state import AgentState

router_llm = get_llm(max_completion_tokens=220)
responder_llm = get_llm(max_completion_tokens=180)

TOOLS = {
    "frappe_create_document": frappe_create_document,
    "frappe_get_list": frappe_get_list,
    "frappe_get_document": frappe_get_document,
    "frappe_get_doctype_definition": frappe_get_doctype_definition,
    "frappe_update_document": frappe_update_document,
    "frappe_delete_document": frappe_delete_document,
}


def build_tool_catalog() -> str:

    blocks = []

    for tool in TOOLS.values():

        try:
            schema = tool.args_schema.model_json_schema()

            params = json.dumps(schema.get("properties", {}), indent=2)

        except Exception:
            params = "{}"

        blocks.append(
            f"{tool.name}\n" f"- {tool.description}\n" f"- Parameters:\n{params}"
        )

    return "\n\n".join(blocks)


TOOL_CATALOG = build_tool_catalog()


ROUTER_PROMPT = """issue 
You are an ERPNext/Frappe assistant.

You must decide whether the user's request is:

CHAT
or
TOOL

CHAT means:
- greetings
- general questions
- capability questions
- questions that don't require accessing Frappe
- missing information

TOOL means:
- list records
- search records
- get a record
- create a record
- update a record
- delete a record

Available tools:

{tool_catalog}

Routing rules:

list, show, find, search, how many
-> frappe_get_list

get a specific document by name
-> frappe_get_document

create, add, make
-> frappe_create_document

update, change, edit, set
-> frappe_update_document

delete, remove, cancel
-> frappe_delete_document

fields, mandatory fields, DocType definition, child table
-> frappe_get_doctype_definition

Return ONLY JSON.

For CHAT, write a helpful, complete answer to the user:

{{
    "route": "chat",
    "answer": "Your natural-language answer here"
}}

SCOPE RULES:

You are only allowed to help with:
- Frappe Framework
- ERPNext
- The approved Frappe DocTypes and their records

If the question is outside that scope, return exactly:

{{
    "route": "chat",
    "answer": "I can help only with Frappe, ERPNext, and the approved business DocTypes."
}}

Do not answer questions about general knowledge, coding outside Frappe,
politics, entertainment, personal advice, or any unrelated topic.


For TOOL:

{{
    "route": "tool",
    "name": "tool_name",
    "parameters": {{}}
}}

Never invent missing business information.

Never invent API credentials.

Today's date is {today}.

If a Sales Order delivery date is provided but transaction date
is not provided, use today's date as transaction_date.

For child tables such as items, use a JSON array.
"""


def router_node(state: AgentState) -> AgentState:

    user_message = state["user_message"]

    prompt = ROUTER_PROMPT.format(
        tool_catalog=TOOL_CATALOG,
        today=date.today().isoformat(),
    )

    response = router_llm.invoke(
        [
            {"role": "system", "content": prompt},
            {
                "role": "system",
                "content": state.get("conversation_context", ""),
            },
            {"role": "user", "content": user_message},
        ]
    )

    content = response.content.strip()

    print("\n========== ROUTER ==========")
    print(content)

    try:

        if content.startswith("```"):

            lines = content.splitlines()

            if lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        result = json.loads(content)

    except Exception:

        return {
            **state,
            "route": "chat",
            "answer": content,
        }

    route = result.get("route", "chat")

    if route == "chat":
        answer = result.get("answer", "").strip()

        if not answer:
            answer = (
                "I can help with general Frappe or ERPNext questions, "
                "or work with records when you provide the required details."
            )

        return {
            **state,
            "route": "chat",
            "answer": answer,
            "tool_executed": False,
        }

    if route != "tool":
        return {
            **state,
            "route": "chat",
            "answer": "I could not determine how to handle that request.",
            "tool_executed": False,
        }

    return {
        **state,
        "route": "tool",
        "tool_name": result.get("name", ""),
        "tool_parameters": result.get("parameters", {}),
    }


def validation_node(state: AgentState) -> AgentState:

    if state.get("route") != "tool":

        return state

    tool_name = state.get("tool_name")

    parameters = state.get("tool_parameters", {})

    if not tool_name:

        return {**state, "validation_error": "No tool was selected."}

    if tool_name not in TOOLS:

        return {
            **state,
            "validation_error": ("The requested operation is not available."),
        }

    
    if not parameters:

        return {**state, "validation_error": ("Required tool parameters are missing.")}

    return {
        **state,
        "validation_error": None,
    }


def resolve_master_links_node(state: AgentState) -> AgentState:
    """Resolve static Frappe Link values before a write reaches Frappe."""
    if state.get("tool_name") not in {
        "frappe_create_document",
        "frappe_update_document",
    }:
        return state

    parameters = state.get("tool_parameters", {})
    doctype = parameters.get("doctype")
    document = parameters.get("document")

    if not isinstance(doctype, str) or not doctype.strip():
        return {**state, "validation_error": "A valid DocType is required."}

    try:
        resolved_document, resolutions = resolve_document_master_links(
            doctype,
            document,
        )
    except MasterLinkResolutionError as exc:
        return {**state, "validation_error": str(exc)}

    resolved_parameters = {**parameters, "document": resolved_document}
    if resolutions:
        print("\n========== MASTER LINK RESOLUTION ==========")
        for resolution in resolutions:
            print(
                f"{resolution['field']}: "
                f"{resolution['supplied']} -> {resolution['canonical']}"
            )

    return {
        **state,
        "tool_parameters": resolved_parameters,
        "validation_error": None,
    }


def approval_node(state: AgentState) -> AgentState:

    tool_name = state.get("tool_name")

    write_tools = {
        "frappe_create_document",
    }

    if tool_name in write_tools:

        print("\n========== APPROVAL REQUIRED ==========")

        # For now this is automatic.
        # Later we will replace this with LangGraph interrupt().
        approved = True

        return {
            **state,
            "requires_approval": True,
            "approved": approved,
        }

    return {
        **state,
        "requires_approval": False,
        "approved": True,
    }


def execute_tool_node(state: AgentState) -> AgentState:

    tool_name = state["tool_name"]

    parameters = state.get("tool_parameters", {})

    tool = TOOLS.get(tool_name)

    if tool is None:

        return {
            **state,
            "tool_executed": False,
            "tool_result": {"success": False, "error": "Unknown tool."},
        }

    print("\n========== TOOL EXECUTION ==========")
    print("TOOL:", tool_name)
    print("PARAMETERS:", parameters)

    try:

        result = tool.invoke(parameters)

    except Exception as exc:

        print("TOOL ERROR:", str(exc))

        result = {
            "success": False,
            "error": str(exc),
        }

    print("\n========== TOOL RESULT ==========")
    print(result)

    return {
        **state,
        "tool_executed": True,
        "tool_result": result,
    }


def error_node(state: AgentState) -> AgentState:

    error = state.get("validation_error", "Something went wrong.")

    return {
        **state,
        "answer": error,
        "tool_executed": False,
    }


RESPONDER_PROMPT = """
Conversation context:

{{state.get("conversation_context", "")}}

User request:

{{state["user_message"]}}

Tool:

{{state.get("tool_name")}}

Tool result:

{{json.dumps(tool_result)}}


You are an ERPNext/Frappe assistant.

You just executed an operation on behalf of the user.

Write a short natural-language response.

Rules:

- Never show raw JSON.
- Never show stack traces.
- Never show exception class names.
- Never expose credentials.
- If successful, explain what happened.
- If failed, explain the actual reason.
- Do not invent information.
- Keep the response concise.
"""


def responder_node(state: AgentState) -> AgentState:
    if state.get("route") == "chat" or state.get("validation_error"):
        return state

    tool_result = state.get("tool_result", {})

    prompt = f"""
Summarize this Frappe operation in at most 80 words.
Do not show JSON, stack traces, credentials, or internal field names.

User request: {state["user_message"]}
Tool: {state.get("tool_name")}
Result: {json.dumps(tool_result, ensure_ascii=False)[:5000]}
"""

    response = responder_llm.invoke(
        [
            {"role": "system", "content": prompt},
            {
                "role": "system",
                "content": state.get("conversation_context", ""),
            },
            {"role": "user", "content": state["user_message"]},
        ]
    )

    return {
        **state,
        "answer": response.content.strip(),
    }


def route_after_router(state: AgentState) -> str:

    if state.get("route") == "chat":

        return "responder"

    return "validate"


def route_after_validation(state: AgentState) -> str:

    if state.get("validation_error"):

        return "error"

    return "approval"


def route_after_master_link_resolution(state: AgentState) -> str:
    if state.get("validation_error"):
        return "error"

    return "approval"


def route_after_approval(state: AgentState) -> str:

    if not state.get("approved", False):

        return "error"

    return "execute"
