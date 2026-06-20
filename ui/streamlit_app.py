"""Streamlit demo UI for the Production RAG API.

A minimal chat front-end that calls the running API's streaming endpoint so
tokens render live, then shows the cited sources and the per-query token/cost
usage. Intended for a quick local demo or a Hugging Face Space.

Run:
    pip install -e ".[ui]"
    streamlit run ui/streamlit_app.py
    # point it at your API (default http://localhost:8000) in the sidebar
"""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

st.set_page_config(page_title="Production RAG", page_icon=":mag:", layout="wide")

# ── Sidebar: connection + retrieval controls ───────────
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", os.getenv("RAG_API_URL", "http://localhost:8000")).rstrip("/")
    api_key = st.text_input("API key (optional)", type="password")
    top_k = st.slider("top_k", min_value=1, max_value=20, value=5)
    stream = st.checkbox("Stream tokens", value=True)
    st.caption("Set API key only if the server has API_KEY_HASH configured.")


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def stream_events(question: str):
    """Yield NDJSON events from POST /chat/stream."""
    with httpx.stream(
        "POST",
        f"{api_url}/chat/stream",
        json={"question": question, "top_k": top_k},
        headers=_headers(),
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.strip():
                yield json.loads(line)


def fetch_answer(question: str) -> dict:
    """Single-shot POST /chat."""
    response = httpx.post(
        f"{api_url}/chat",
        json={"question": question, "top_k": top_k},
        headers=_headers(),
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for s in sources:
            meta = s.get("metadata", {})
            citation = meta.get("citation", "?")
            source = meta.get("source", "unknown")
            st.markdown(f"**[{citation}]** `{source}`")
            st.caption(s.get("content", "")[:500])


def render_usage(usage: dict, latency_ms: float | None) -> None:
    cols = st.columns(4)
    cols[0].metric("Input tokens", usage.get("input_tokens", "-"))
    cols[1].metric("Output tokens", usage.get("output_tokens", "-"))
    cost = usage.get("cost_usd")
    cols[2].metric("Est. cost", f"${cost:.5f}" if isinstance(cost, (int, float)) else "-")
    cols[3].metric("Latency", f"{latency_ms:.0f} ms" if latency_ms else "-")


# ── Main chat ──────────────────────────────────────────
st.title("Production RAG")
st.caption("Hybrid retrieval + reranking + GraphRAG, with grounded, cited answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the ingested documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if stream:
                placeholder = st.empty()
                answer, sources, usage, latency = "", [], {}, None
                for event in stream_events(question):
                    kind = event.get("event")
                    if kind == "sources":
                        sources = event.get("sources", [])
                        latency = event.get("latency_ms")
                    elif kind == "token":
                        answer += event.get("token", "")
                        placeholder.markdown(answer + "█")
                    elif kind == "done":
                        answer = event.get("answer", answer)
                        usage = event.get("usage", {})
                        latency = event.get("latency_ms", latency)
                    elif kind == "error":
                        st.error(event.get("detail", "stream error"))
                placeholder.markdown(answer)
            else:
                result = fetch_answer(question)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                usage = result.get("usage", {})
                latency = result.get("latency_ms")
                st.markdown(answer)

            render_sources(sources)
            render_usage(usage, latency)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
