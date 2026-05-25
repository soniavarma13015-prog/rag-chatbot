import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import chat, health, ingest
from backend.dependencies import get_vector_store_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Chatbot API…")
    vsm = get_vector_store_manager()
    await asyncio.get_event_loop().run_in_executor(None, vsm.ensure_collection)
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down RAG Chatbot API.")


app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(health.router)


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(str(INDEX_HTML))