
import logging
import os
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph.graph import graph


app = FastAPI(
    title="LangGraph Frappe Agent",
    version="3.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

_THREAD_COOKIE = "frappe_assistant_thread"
_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/")
def chat_page():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "LangGraph + Frappe",
    }


def _thread_id(http_request: Request) -> tuple[str, bool]:
    supplied_thread_id = http_request.cookies.get(_THREAD_COOKIE)
    if supplied_thread_id:
        try:
            return str(UUID(supplied_thread_id)), False
        except ValueError:
            pass
    return str(uuid4()), True


@app.post("/ask")
async def ask(chat_request: AskRequest, http_request: Request, response: Response):
    user_message = chat_request.message.strip()
    thread_id, is_new_thread = _thread_id(http_request)

    try:
        final_state = await graph.ainvoke(
            {"user_message": user_message},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception:
        logger.exception("POST /ask failed")
        raise HTTPException(
            status_code=500,
            detail="The assistant could not process this request.",
        )
    answer = final_state.get("answer", "").strip()

    if is_new_thread:
        response.set_cookie(
            key=_THREAD_COOKIE,
            value=thread_id,
            max_age=60 * 60 * 24 * 7,
            httponly=True,
            samesite="lax",
            secure=_COOKIE_SECURE,
        )

    return {
        "answer": answer,
        "tool_executed": final_state.get("tool_executed", False),
        "tool_name": final_state.get("tool_name"),
        "presentation": final_state.get("presentation"),
    }
