import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.pipeline import query_pipeline, stream_query
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
    usage: dict = {}


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
        usage=result.get("usage", {}),
    )


@router.post("/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest, _key=Depends(verify_api_key)):
    """Token-by-token streaming as newline-delimited JSON.

    Event sequence: one ``sources`` event, then a ``token`` event per
    generated token, then a terminal ``done`` event with the assembled
    answer and token/cost usage. Errors surface as an ``error`` event.
    """
    async def event_generator():
        try:
            async for event in stream_query(body.question, body.top_k):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("Streaming chat error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
