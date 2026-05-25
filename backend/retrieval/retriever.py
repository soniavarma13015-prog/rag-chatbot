import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from backend.config import Config
from backend.chat.schemas import SourceDoc
from backend.ingestion.embedder import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, vsm: VectorStoreManager, cfg: Config) -> None:
        self.vsm = vsm
        self.cfg = cfg
        self._langchain_store = None

    def _get_store(self):
        if self._langchain_store is None:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.cfg.google_api_key,
            )
            self._langchain_store = Chroma(
                client=self.vsm.client,
                collection_name=self.cfg.collection_name,
                embedding_function=embeddings,
            )
        return self._langchain_store

    def retrieve(self, query: str) -> list[SourceDoc]:
        store = self._get_store()
        k = self.cfg.top_k

        logger.info(
            "Retrieving top-%d docs for query (len=%d) use_mmr=%s",
            k, len(query), self.cfg.use_mmr,
        )

        if self.cfg.use_mmr:
            results_with_scores = store.similarity_search_with_relevance_scores(
                query, k=k * 4
            )
            source_docs = [
                SourceDoc(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4),
                )
                for doc, score in results_with_scores
            ]
            source_docs = sorted(source_docs, key=lambda d: d.score, reverse=True)[:k]
        else:
            results_with_scores = store.similarity_search_with_relevance_scores(query, k=k)
            source_docs = [
                SourceDoc(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4),
                )
                for doc, score in results_with_scores
            ]

        # Only return docs above relevance threshold
        filtered = [d for d in source_docs if d.score > 0.5]
        result = filtered if filtered else source_docs[:1]

        # DEBUG - remove after fixing
        for d in result:
            logger.info("RETURNING CHUNK: score=%s content=%s", d.score, d.content[:200])

        return result