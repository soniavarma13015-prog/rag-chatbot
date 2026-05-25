import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> list[Document]:
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_hash"] = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()

    logger.info(
        "Chunked %d source sections into %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks