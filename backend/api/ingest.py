import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.chat.schemas import UploadResponse
from backend.config import config
from backend.dependencies import get_vector_store_manager
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.embedder import VectorStoreManager
from backend.ingestion.loader import SUPPORTED_EXTENSIONS, load_documents

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/ingest", response_model=UploadResponse)
async def ingest_file(
    file: UploadFile,
    vsm: VectorStoreManager = Depends(get_vector_store_manager),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / (file.filename or "upload")
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info("Received upload: %s (%d bytes)", file.filename, len(content))

        docs = await asyncio.get_event_loop().run_in_executor(
            None, load_documents, tmp_path
        )

        # FIX: normalize source to just filename, not full temp path
        for doc in docs:
            doc.metadata["source"] = file.filename or "upload"

        chunks = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chunk_documents(docs, config.chunk_size, config.chunk_overlap),
        )
        inserted = await asyncio.get_event_loop().run_in_executor(
            None, vsm.upsert, chunks
        )

    logger.info("Ingest complete: %s → %d chunks indexed", file.filename, inserted)
    return UploadResponse(
        message=f"Successfully indexed '{file.filename}'.",
        chunks_indexed=inserted,
        file_name=file.filename or "",
    )