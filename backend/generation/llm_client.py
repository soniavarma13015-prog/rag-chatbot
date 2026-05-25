import logging
from typing import AsyncIterator

from langchain_groq import ChatGroq
from backend.config import config as global_config

logger = logging.getLogger(__name__)


async def stream_response(messages: list[dict]) -> AsyncIterator[str]:
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=global_config.groq_api_key,
            streaming=True,
        )
        langchain_messages = []
        for m in messages:
            if m["role"] == "system":
                langchain_messages.append(("system", m["content"]))
            elif m["role"] == "user":
                langchain_messages.append(("human", m["content"]))
            elif m["role"] == "assistant":
                langchain_messages.append(("ai", m["content"]))

        async for chunk in llm.astream(langchain_messages):
            if chunk.content:
                yield chunk.content
        yield "[DONE]"
    except Exception as exc:
        logger.exception("LLM streaming error: %s", exc)
        yield f"[ERROR]: {exc}"
        yield "[DONE]"