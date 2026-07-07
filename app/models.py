# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str = Field(..., min_length=1)
    file_id: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    entity_id: Optional[str] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str = Field(..., min_length=1)
    file_ids: List[str] = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class GlobalRetrievalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    entity_id: Optional[str] = None
    file_id_prefix: Optional[str] = Field(default=None, min_length=1)


class RetrievalSearchResult(BaseModel):
    rank: int
    content: str
    score: float
    score_type: str
    file_id: Optional[str] = None
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    page: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class RetrievalSearchResponse(BaseModel):
    query: str
    file_id: str
    top_k: int
    result_count: int
    results: List[RetrievalSearchResult]


class GlobalRetrievalSearchResponse(BaseModel):
    query: str
    top_k: int
    result_count: int
    candidate_mode: str
    file_id_prefix: Optional[str] = None
    results: List[RetrievalSearchResult]


class RagSource(BaseModel):
    rank: int
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    score: float
    score_type: str
    content: str
    metadata: dict = Field(default_factory=dict)


class RagAnswerStrategy(str, Enum):
    extractive_context_v1 = "extractive_context_v1"
    llm_grounded_context_v1 = "llm_grounded_context_v1"
    no_context_refusal = "no_context_refusal"
    low_confidence_refusal = "low_confidence_refusal"


class RagChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    file_id: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    entity_id: Optional[str] = None
    max_context_chars: int = Field(default=1200, ge=200, le=8000)
    use_llm: Optional[bool] = None


class RagGlobalChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    entity_id: Optional[str] = None
    file_id_prefix: Optional[str] = Field(default=None, min_length=1)
    max_context_chars: int = Field(default=1200, ge=200, le=8000)
    use_llm: Optional[bool] = None


class RagChatResponse(BaseModel):
    query: str
    file_id: str
    answer: str
    refusal: bool
    answer_strategy: RagAnswerStrategy
    used_context_count: int
    sources: List[RagSource]


class RagGlobalChatResponse(BaseModel):
    query: str
    candidate_mode: str
    file_id_prefix: Optional[str] = None
    answer: str
    refusal: bool
    answer_strategy: RagAnswerStrategy
    used_context_count: int
    sources: List[RagSource]
