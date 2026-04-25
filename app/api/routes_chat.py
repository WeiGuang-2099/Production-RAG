from fastapi import APIRouter
from pydantic import BaseModel
from app.core.pipeline import query_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: float


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = query_pipeline(request.question)
    return ChatResponse(**result)
