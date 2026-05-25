import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.chat.history import ChatHistoryManager
from backend.chat.schemas import ChatRequest, SourceDoc
from backend.dependencies import get_history_manager, get_retriever, get_vector_store_manager
from backend.generation.llm_client import stream_response
from backend.generation.prompt_builder import build_prompt
from backend.ingestion.embedder import VectorStoreManager
from backend.retrieval.retriever import RAGRetriever

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    vsm: VectorStoreManager = Depends(get_vector_store_manager),
    retriever: RAGRetriever = Depends(get_retriever),
    history_manager: ChatHistoryManager = Depends(get_history_manager),
) -> StreamingResponse:
    logger.info("Chat request: session=%s query_len=%d", request.session_id, len(request.query))

    info = await asyncio.get_event_loop().run_in_executor(None, vsm.get_collection_info)
    if info["count"] == 0:
        async def empty_stream():
            yield f"data: {json.dumps({'token': 'No documents indexed yet. Please upload a document first.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    context_docs: list[SourceDoc] = await asyncio.get_event_loop().run_in_executor(
        None, retriever.retrieve, request.query
    )
    logger.info("Retrieved %d context docs", len(context_docs))

    history = history_manager.get_history(request.session_id)
    messages = build_prompt(request.query, context_docs, history)

    async def event_generator():
        full_response: list[str] = []
        try:
            async for token in stream_response(messages):
                if token == "[DONE]":
                    assistant_msg = "".join(full_response)
                    history_manager.add_turn(request.session_id, request.query, assistant_msg)
                    sources_payload = [doc.model_dump() for doc in context_docs]
                    yield f"data: {json.dumps({'done': True, 'sources': sources_payload})}\n\n"
                    return
                elif token.startswith("[ERROR]:"):
                    yield f"data: {json.dumps({'error': token})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
                    return
                else:
                    full_response.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")