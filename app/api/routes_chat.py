import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.pipeline import query_pipeline
from app.api.deps import verify_api_key, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)


class SourceItem(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: float
    total_sources: int


@router.post("", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest, _key=Depends(verify_api_key)):
    result = await asyncio.to_thread(query_pipeline, body.question, body.top_k)
    sources = result.get("sources", [])
    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        latency_ms=result["latency_ms"],
        total_sources=len(sources),
    )


@router.post("/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest, _key=Depends(verify_api_key)):
    async def event_generator():
        try:
            result = await asyncio.to_thread(query_pipeline, body.question, body.top_k)
            # Emit sources first
            yield json.dumps({
                "event": "retrieval_complete",
                "sources": result.get("sources", []),
                "latency_ms": result["latency_ms"],
            }) + "\n"
            # Emit answer
            yield json.dumps({
                "event": "answer_complete",
                "answer": result["answer"],
            }) + "\n"
            # Done
            yield json.dumps({"event": "complete"}) + "\n"
        except Exception as exc:
            logger.error("Streaming chat error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
