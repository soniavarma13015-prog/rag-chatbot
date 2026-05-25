from pydantic_settings import BaseSettings
from chromadb.config import Settings as ChromaSettings


class Config(BaseSettings):
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_docs"
    chunk_size: int = 200
    chunk_overlap: int = 20
    top_k: int = 1
    use_mmr: bool = True
    llm_model: str = "llama3-8b-8192"
    embed_model: str = "models/gemini-embedding-001"
    max_history_turns: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def chroma_settings(self) -> ChromaSettings:
        return ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        )


config = Config()