from pydantic import BaseModel


class SourceDoc(BaseModel):
    content: str
    source: str
    page: int | None = None
    score: float


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    session_id: str


class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int
    file_name: str


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    doc_count: int