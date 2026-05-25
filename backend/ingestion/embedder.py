import logging

import chromadb
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config import Config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=cfg.google_api_key,
        )
        self.client = chromadb.PersistentClient(
            path=cfg.chroma_persist_dir,
            settings=cfg.chroma_settings,
        )
        self.collection = None

    def ensure_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        info = self.get_collection_info()
        logger.info(
            "Collection '%s' ready — %d documents indexed",
            info["name"],
            info["count"],
        )

    def _get_collection(self):
        if self.collection is None:
            self.ensure_collection()
        return self.collection

    def upsert(self, docs: list[Document]) -> int:
        collection = self._get_collection()

        # FIX: delete old chunks for this source file before upserting
        if docs:
            source = docs[0].metadata.get("source", "")
            if source:
                existing = collection.get(where={"source": source}, include=["metadatas"])
                old_ids = existing.get("ids", [])
                if old_ids:
                    collection.delete(ids=old_ids)
                    logger.info("Deleted %d old chunks for '%s'", len(old_ids), source)

        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        ids = [d.metadata["chunk_hash"] for d in docs]

        vectors = self.embeddings.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Upserted %d new chunks into '%s'", len(docs), self.cfg.collection_name)
        return len(docs)

    def get_collection_info(self) -> dict:
        collection = self._get_collection()
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata,
        }

    def reset_collection(self) -> None:
        self.client.delete_collection(self.cfg.collection_name)
        self.collection = None
        self.ensure_collection()
        logger.info("Collection '%s' has been reset", self.cfg.collection_name)