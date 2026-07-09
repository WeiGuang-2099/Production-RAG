import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent, stream_agent
from app.api.deps import limiter, verify_api_key
from app.guardrails.service import apply_output, check_input

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    sources: list[str] | None = None


class AgentResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    usage: dict = {}
    route: str = ""
    attempts: int = 0
    guardrails: dict = {}


@router.post("", response_model=AgentResponse)
@limiter.limit("30/minute")
async def agent(request: Request, body: AgentRequest, _key=Depends(verify_api_key)):
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked})
    result = await asyncio.to_thread(run_agent, body.question, body.top_k, body.sources)
    guarded = apply_output(result["answer"])
    return AgentResponse(
        answer=guarded["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        usage=result.get("usage", {}),
        route=result.get("route", ""),
        attempts=result.get("attempts", 0),
        guardrails={"pii_redacted": guarded["pii_redacted"], "flags": guarded["flags"]},
    )


@router.post("/stream")
@limiter.limit("30/minute")
async def agent_stream(request: Request, body: AgentRequest, _key=Depends(verify_api_key)):
    blocked = check_input(body.question)
    if blocked:
        raise HTTPException(status_code=400, detail={"error": "blocked by input guardrails", "patterns": blocked})

    async def event_generator():
        try:
            async for event in stream_agent(body.question, body.top_k, body.sources):
                if event.get("event") == "done":
                    g = apply_output(event.get("answer", ""))
                    event["answer"] = g["answer"]
                    event["guardrails"] = {"pii_redacted": g["pii_redacted"], "flags": g["flags"]}
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("agent stream error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
