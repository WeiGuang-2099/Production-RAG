import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import limiter, verify_api_key
from app.core.pipeline import query_pipeline, stream_query
from app.guardrails.service import apply_output, check_input

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    sources: list[str] | None = None
    history: list[HistoryTurn] | None = None


class SourceItem(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: float
    total_sources: int
    usage: dict = {}
    guardrails: dict = {}
    condensed_question: str | None = None


@router.post("", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest, _key=Depends(verify_api_key)):
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked})
    history = [t.model_dump() for t in body.history or []]
    result = await asyncio.to_thread(query_pipeline, body.question, body.top_k, body.sources, history)
    guarded = apply_output(result["answer"])
    sources = result.get("sources", [])
    return ChatResponse(
        answer=guarded["answer"],
        sources=sources,
        latency_ms=result["latency_ms"],
        total_sources=len(sources),
        usage=result.get("usage", {}),
        guardrails={"pii_redacted": guarded["pii_redacted"], "flags": guarded["flags"]},
        condensed_question=result.get("condensed_question"),
    )


@router.post("/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest, _key=Depends(verify_api_key)):
    """Token-by-token streaming as newline-delimited JSON.

    Event sequence: one ``sources`` event, then a ``token`` event per
    generated token, then a terminal ``done`` event with the assembled
    answer and token/cost usage. Errors surface as an ``error`` event.

    Guardrails: input injection is blocked before streaming starts; PII
    redaction/toxicity flagging is applied to the final ``done`` answer
    only (tokens stream raw).
    """
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked})

    history = [t.model_dump() for t in body.history or []]

    async def event_generator():
        try:
            async for event in stream_query(body.question, body.top_k, body.sources, history):
                if event.get("event") == "done":
                    g = apply_output(event.get("answer", ""))
                    event["answer"] = g["answer"]
                    event["guardrails"] = {"pii_redacted": g["pii_redacted"], "flags": g["flags"]}
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("Streaming chat error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
