import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent, stream_agent
from app.api.deps import limiter, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)


class AgentResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    usage: dict = {}
    route: str = ""
    attempts: int = 0


@router.post("", response_model=AgentResponse)
@limiter.limit("30/minute")
async def agent(request: Request, body: AgentRequest, _key=Depends(verify_api_key)):
    result = await asyncio.to_thread(run_agent, body.question, body.top_k)
    return AgentResponse(**result)


@router.post("/stream")
@limiter.limit("30/minute")
async def agent_stream(request: Request, body: AgentRequest, _key=Depends(verify_api_key)):
    async def event_generator():
        try:
            async for event in stream_agent(body.question, body.top_k):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("agent stream error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
