import json

from app.llm import get_llm
from app.tools.frappe import (
    frappe_create_document,
    frappe_get_list,
    frappe_get_document,
    # frappe_update_document,
    # frappe_delete_document,
)
from datetime import date


# ---------------------------------------------------------------------
# Tool registry — add/remove a tool ONLY here. The prompt catalog below
# and the dispatcher both read from this dict dynamically.
# ---------------------------------------------------------------------
TOOLS = {
    "frappe_create_document": frappe_create_document,
    "frappe_get_list": frappe_get_list,
    "frappe_get_document": frappe_get_document,
    # "frappe_update_document": frappe_update_document,
    # "frappe_delete_document": frappe_delete_document,
}


def _build_tool_catalog() -> str:
    """
    Renders the system-prompt tool catalog straight from each registered
    tool's name/description/args_schema. Register a new @tool in
    app/tools and it appears here automatically — no prompt edits needed.
    """
    blocks = []

    for tool in TOOLS.values():
        try:
            schema = tool.args_schema.model_json_schema()
            params = json.dumps(schema.get("properties", {}), indent=2)
        except Exception:
            params = "{}"

        blocks.append(
            f"{tool.name}\n"
            f"- {tool.description}\n"
            f"- Parameters schema:\n{params}"
        )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------
# STAGE 1 — ROUTER
# Decides: is this a real data request (needs a tool) or a conversational
# / meta / capability question (answer directly, no tool, no JSON)?
# ---------------------------------------------------------------------
ROUTER_PROMPT_TEMPLATE = """
You are an ERPNext/Frappe assistant with two modes of replying:

MODE A — CHAT (plain natural language, no JSON at all):
Use this whenever the user is:
- Greeting you, making small talk, or asking what you can do /
  what you are / how to use you.
- Asking a general question about Frappe/ERPNext concepts that does
  NOT require reading or changing any specific record in this system.
- Missing information you'd need to safely call a tool (e.g. they want
  to create something but haven't said which doctype or what fields).
  In this case, ask them a short clarifying question in plain text.
In MODE A you must reply with ordinary, friendly sentences. Never
output JSON in this mode.

MODE B — TOOL CALL (ONLY valid JSON, nothing else):
Use this ONLY when the user is clearly asking you to read, list,
create, update, or delete an actual record in this system, and you
have enough information to do it.

Available tools:

{tool_catalog}

Routing rules:
- "list", "show", "find", "how many", "get all", "search" -> READ.
  Use frappe_get_list (many records) or frappe_get_document (one
  record by name/ID). NEVER use frappe_create_document for these.
- "create", "add", "make a new" -> frappe_create_document.
- "update", "change", "edit", "set" on an existing record ->
  frappe_update_document.
- "delete", "remove", "cancel" a specific record ->
  frappe_delete_document.
- General/capability/greeting questions -> MODE A. Do not call a tool
  just because Frappe was mentioned.



{{
  "name": "tool_name",
  "parameters": {{
    ...
  }}
}}

IMPORTANT (applies to both modes):
- Never invent API credentials or expose credentials.
- Never claim an operation succeeded unless a tool actually returned
  a successful response.
- Do not invent missing business information.
- For child-table fields such as "items", parameters must use a JSON
  array, NOT a JSON string.

Today's date is {today}. If a Sales Order delivery date is provided
but no Sales Order date is given, use today's date as the
transaction_date.
"""


# ---------------------------------------------------------------------
# STAGE 2 — RESPONDER
# Takes the raw tool result (success data OR error) and turns it into
# a natural, human sentence. No stack traces, no raw JSON dumps.
# ---------------------------------------------------------------------
RESPONDER_SYSTEM_PROMPT = """
You just executed a Frappe/ERPNext action on the user's behalf. You are
given the user's original request and the raw result (JSON) from the
system. Write a short, natural, human reply summarizing it — the way a
helpful colleague would explain it out loud, not the way an API would.

Rules:
- Never show raw JSON, field names like "success"/"status_code", stack
  traces, file paths, or exception class names.
- If the result succeeded and contains a list, summarize it naturally
  (e.g. name a few records, mention the count if it's long). If it's a
  single record, describe the useful fields in a sentence or two.
- If the result failed, figure out the actual root cause from the error
  text (e.g. a missing/invalid field, a permission issue, a duplicate)
  and explain it in one or two plain sentences — what went wrong and
  what the user should do about it. If the underlying cause truly can't
  be determined from the error text, say so plainly and suggest they
  check the record's required fields rather than guessing.
- Keep it concise. No headers, no bullet-point dumps of raw data unless
  that's genuinely the clearest way to present a short list.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


def get_agent():
    return DatabaseAgent()


class DatabaseAgent:

    def __init__(self):
        self.llm = get_llm()
        self.tool_catalog = _build_tool_catalog()

    def _humanize(self, user_message: str, tool_name: str, tool_result: dict) -> str:
        prompt = (
            f"User's original request: {user_message}\n\n"
            f"Tool used: {tool_name}\n\n"
            f"Raw result:\n{json.dumps(tool_result)}\n\n"
            "Write the natural-language reply now."
        )

        messages = [
            {"role": "system", "content": RESPONDER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()

    def invoke(self, user_message: str):
        today = date.today().isoformat()

        router_prompt = ROUTER_PROMPT_TEMPLATE.format(
            tool_catalog=self.tool_catalog,
            today=today,
        )

        messages = [
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": user_message},
        ]
        response = self.llm.invoke(messages)

        content = response.content

        print("========================================")
        print("ROUTER RESPONSE")
        print("========================================")
        print(content)

        try:
            tool_request = _extract_json(content)

        except json.JSONDecodeError:
            # MODE A — plain-language reply, no tool involved.
            return {
                "answer": content.strip(),
                "tool_executed": False,
            }

        tool_name = tool_request.get("name")
        parameters = tool_request.get("parameters", {})

        print("\n========================================")
        print("TOOL REQUEST")
        print("========================================")
        print("NAME:", tool_name)
        print("PARAMETERS:", parameters)

        tool = TOOLS.get(tool_name)

        if tool is None:
            # The router hallucinated a tool name that isn't registered.
            # Don't show that internally to the user — ask them to
            # rephrase instead of leaking an "unknown tool" error.
            return {
                "answer": (
                    "I couldn't quite match that to something I can do. "
                    "Could you rephrase what you'd like me to look up, "
                    "create, update, or delete?"
                ),
                "tool_executed": False,
            }

        print("\n========================================")
        print("EXECUTING TOOL")
        print("========================================")

        try:
            tool_result = tool.invoke(parameters)

        except Exception as exc:
            print("TOOL ERROR:", str(exc))
            tool_result = {"success": False, "error": str(exc)}

        print("\n========================================")
        print("TOOL RESULT")
        print("========================================")
        print(tool_result)

        human_answer = self._humanize(user_message, tool_name, tool_result)

        return {
            "answer": human_answer,
            "tool_executed": True,
            "tool_name": tool_name,
            "raw_result": tool_result,
        }