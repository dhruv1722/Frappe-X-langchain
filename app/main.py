
import logging

logger = logging.getLogger(__name__)


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph.graph import graph
from app.memory import MemoryManager


app = FastAPI(
    title="LangGraph Frappe Agent",
    version="3.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

memory = MemoryManager()


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=100)
    conversation_id: str | None = Field(default=None, max_length=100)


@app.get("/")
def chat_page():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "LangGraph + Frappe",
    }


@app.post("/ask")
def ask(request: AskRequest):
    user_message = request.message.strip()

    conversation_id = memory.ensure_conversation(
        user_id=request.user_id,
        conversation_id=request.conversation_id,
    )

    forget_response = memory.forget_memory(
        request.user_id,
        user_message,
    )
    if forget_response:
        memory.save_message(conversation_id, "user", user_message)
        memory.save_message(conversation_id, "assistant", forget_response)

        return {
            "answer": forget_response,
            "conversation_id": conversation_id,
            "tool_executed": False,
            "tool_name": None,
        }

    saved_memory_response = memory.save_explicit_memory(
        request.user_id,
        user_message,
    )
    
    conversation_context = memory.build_context(
        request.user_id,
        conversation_id,
    )

    memory.save_message(conversation_id, "user", user_message)

    initial_state = {
        "user_message": user_message,
         "conversation_context": conversation_context,
        "conversation_context": memory.build_context(
            request.user_id,
            conversation_id,
        ),
        "today": "",
        "route": "",
        "tool_name": "",
        "tool_parameters": {},
        "tool_executed": False,
        "tool_result": {},
        "validation_error": None,
        "answer": "",
        "requires_approval": False,
        "approved": False,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception:
        logger.exception("POST /ask failed")
        raise HTTPException(
            status_code=500,
            detail="The assistant could not process this request.",
        )
    answer = final_state.get("answer", "").strip()

    if saved_memory_response:
        answer = f"{saved_memory_response}\n\n{answer}"

    memory.save_message(conversation_id, "assistant", answer)

    return {
        "answer": answer,
        "conversation_id": conversation_id,
        "tool_executed": final_state.get("tool_executed", False),
        "tool_name": final_state.get("tool_name"),
    }