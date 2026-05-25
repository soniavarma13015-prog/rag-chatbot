import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _load_single_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": path.name,
                "file_name": path.name,
                "file_type": "pdf",
            })
        return docs

    if ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": str(path),
                "file_name": path.name,
                "file_type": ext.lstrip("."),
                "page": 0,
            })
        return docs

    if ext == ".docx":
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": str(path),
                "file_name": path.name,
                "file_type": "docx",
                "page": 0,
            })
        return docs

    logger.warning("Unsupported file extension: %s — skipping %s", ext, path.name)
    return []


def load_documents(source: str | Path) -> list[Document]:
    source = Path(source)
    documents: list[Document] = []

    if source.is_dir():
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in source.rglob(f"*{ext}"):
                logger.info("Loading file: %s", file_path)
                documents.extend(_load_single_file(file_path))
    elif source.is_file():
        logger.info("Loading file: %s", source)
        documents.extend(_load_single_file(source))
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    logger.info("Loaded %d document sections from %s", len(documents), source)
    return documents