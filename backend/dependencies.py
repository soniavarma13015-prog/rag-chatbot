from backend.config import config
from backend.ingestion.embedder import VectorStoreManager
from backend.retrieval.retriever import RAGRetriever
from backend.chat.history import ChatHistoryManager

_vsm = None
_retriever = None
_history_manager = None


def get_vector_store_manager() -> VectorStoreManager:
    global _vsm
    if _vsm is None:
        _vsm = VectorStoreManager(config)
    return _vsm


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(get_vector_store_manager(), config)
    return _retriever


def get_history_manager() -> ChatHistoryManager:
    global _history_manager
    if _history_manager is None:
        _history_manager = ChatHistoryManager()
    return _history_manager