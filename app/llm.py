import os
from langchain_openai import ChatOpenAI


def get_llm(max_completion_tokens: int = 2080):
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        # model="groq/compound-mini",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        temperature=0,
        top_p=0.95,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort="low",
        timeout=30,
        max_retries=2,
    )
