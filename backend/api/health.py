import asyncio
import logging

from fastapi import APIRouter, Depends

from backend.chat.schemas import HealthResponse
from backend.dependencies import get_vector_store_manager
from backend.ingestion.embedder import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health_check(vsm: VectorStoreManager = Depends(get_vector_store_manager)) -> HealthResponse:
    info = await asyncio.get_event_loop().run_in_executor(None, vsm.get_collection_info)
    return HealthResponse(
        status="ok",
        collection_name=info["name"],
        doc_count=info["count"],
    )