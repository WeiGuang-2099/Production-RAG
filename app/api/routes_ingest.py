from fastapi import APIRouter
from pydantic import BaseModel
from app.core.pipeline import ingest_pipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    source: str


class IngestResponse(BaseModel):
    source: str
    chunks: int


@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    result = ingest_pipeline(request.source)
    return IngestResponse(**result)
