from fastapi import FastAPI
from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.observability.tracing import setup_tracing

setup_tracing()

app = FastAPI(title="Production RAG", version="0.1.0")
app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
