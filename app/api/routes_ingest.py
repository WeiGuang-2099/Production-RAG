import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator

from app.api.deps import limiter, verify_api_key
from app.config import get_settings
from app.core.pipeline import ingest_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    source: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        from app.ingestion.validation import validate_source as _validate
        return _validate(v, get_settings())


class IngestResponse(BaseModel):
    source: str
    chunks: int
    status: str = "ingested"


@router.post("", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest(request: Request, body: IngestRequest, _key=Depends(verify_api_key)):
    result = await asyncio.to_thread(ingest_pipeline, body.source)
    return IngestResponse(**result)


@router.get("/documents")
async def list_documents(_key=Depends(verify_api_key)):
    """List all ingested document records."""
    settings = get_settings()
    tracking_file = Path(settings.DATA_DIR) / "ingestions.json"
    if not tracking_file.exists():
        return []
    try:
        with open(tracking_file) as f:
            data = json.load(f)
        return [
            {"id": k, "source": v["source"], "chunks": v["chunks"], "ingested_at": v.get("ingested_at", "")}
            for k, v in data.items()
        ]
    except Exception as exc:
        logger.error("Failed to read ingestion tracking: %s", exc)
        return []


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, _key=Depends(verify_api_key)):
    """Remove a document record from tracking."""
    settings = get_settings()
    tracking_file = Path(settings.DATA_DIR) / "ingestions.json"
    if not tracking_file.exists():
        raise HTTPException(status_code=404, detail="No ingestion records found")
    try:
        with open(tracking_file) as f:
            data = json.load(f)
        if doc_id not in data:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        removed = data.pop(doc_id)
        with open(tracking_file, "w") as f:
            json.dump(data, f, indent=2)
        return {"status": "deleted", "doc_id": doc_id, "source": removed["source"]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete document record: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
