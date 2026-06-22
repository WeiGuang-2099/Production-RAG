# React Frontend — Design

**Status:** approved (design phase)
**Date:** 2026-06-22
**Scope:** Replace the minimal Streamlit demo with a single, polished React SPA that exposes
**every** backend capability through a UI entry point. One new backend endpoint
(`POST /ingest/upload`) is added; all other endpoints are unchanged.

## Goal

The current demo UI (`ui/streamlit_app.py`) only exposes chat. Document ingestion, listing, and
deletion have no UI entry point, and agent route / self-correction / guardrails metadata is not
surfaced. For an interview portfolio, a crude UI that hides most of the system undersells the
backend engineering.

Build a React single-page app that:

1. Gives **every** backend endpoint and every meaningful response field a visible entry point
   (no "built it but you can't reach it from the UI").
2. Looks intentional and modern, with tasteful motion.
3. Deploys as a static site that talks directly to the existing FastAPI backend.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Tech direction | React SPA (Vite + React + TypeScript), not Streamlit, not Next.js |
| Ingestion | Drag-drop file upload **and** URL input; add `POST /ingest/upload` to the backend |
| Deploy / auth | Static SPA, public demo with backend auth **off** (`API_KEY_HASH` unset); rely on existing rate limiting |
| Streamlit | **Delete it.** One frontend only. |
| Directory | `frontend/` |
| About page | Yes — architecture showcase (portfolio value) |
| Palette | `#eaebef` (bg), `#a1a8ae` (muted), `#09568c` (primary), + a derived dark ink for body text, + one semantic red for errors |
| Motion | framer-motion, restrained |

## No-orphan-feature map

Every backend capability maps to a UI entry point. This table is the acceptance contract.

| Backend capability | Endpoint | Frontend entry point |
|---|---|---|
| Plain Q&A | `POST /chat` | Chat page (stream toggle OFF) |
| Streaming Q&A | `POST /chat/stream` | Chat page (stream toggle ON, default) |
| Agentic Q&A | `POST /agent` | Chat page (Agent mode + stream OFF) |
| Streaming agentic Q&A | `POST /agent/stream` | Chat page (Agent mode + stream ON) + reasoning trace |
| File upload ingest | `POST /ingest/upload` (NEW) | Documents page — drag-drop zone |
| URL ingest | `POST /ingest` | Documents page — URL input |
| List documents | `GET /ingest/documents` | Documents page — table |
| Delete document | `DELETE /ingest/documents/{id}` | Documents page — per-row delete |
| Liveness / readiness | `GET /health/live`, `/health/ready` | Header status indicator (dot + dependency tooltip) |
| Token / cost usage | response `usage` | Chat usage bar |
| Guardrails (PII redact / flags) | response `guardrails` | Chat badge |
| Retrieval sources + scores | response `sources` | Chat source cards |
| Agent route + attempts | response `route`, `attempts` | Chat trace + answer header |
| System architecture | README mermaid | About page |

## Architecture

### Stack & layout
- **Vite + React + TypeScript**, **Tailwind CSS**, `react-router` (3 routes), `react-markdown`
  (answers + source snippets), `lucide-react` (icons), `framer-motion` (motion).
- No global state library. React hooks plus a small `SettingsContext` holding the API base URL
  and an optional API key (key field hidden by default since the public demo runs auth-off, but
  present so the same build works against an auth-on backend).
- New top-level `frontend/` directory, fully isolated from the Python package.

### Talking to the backend
- The SPA calls FastAPI directly. CORS is already configured from the `CORS_ORIGINS` setting;
  add `http://localhost:5173` (Vite dev) to `.env.example` and document setting it to the deployed
  origin in production.
- API base URL from `VITE_API_URL` at build time, overridable at runtime via the Settings panel.
- A single `apiClient` module wraps fetch: injects `Authorization` when a key is set, parses JSON,
  raises a typed `ApiError` on non-2xx (carrying status + detail).

### Streaming
- `/chat/stream` and `/agent/stream` return newline-delimited JSON. A `streamNdjson` helper reads
  the `fetch` `ReadableStream`, splits on `\n`, and yields parsed events. Event protocol (unchanged
  from the backend): `step` (agent node), `sources`, `token`, `done` (answer + usage + latency +
  guardrails), `error`. This is the same contract the Streamlit app used.

## Backend change: `POST /ingest/upload`

Added to `app/api/routes_ingest.py`. The only backend code change.

- Accepts `multipart/form-data` with a single `UploadFile`.
- Validation reuses `app/ingestion/validation.ALLOWED_SUFFIXES` (`.pdf/.md/.markdown`) and
  `settings.MAX_FILE_SIZE_MB`. Reject other suffixes (400) and oversize files (413).
- Sanitize the filename to its basename (drop any path components) to prevent traversal, then save
  under `DATA_DIR`. On name collision, suffix with a short uuid.
- Call the existing `ingest_pipeline(saved_path)` and return the existing `IngestResponse`
  (`source`, `chunks`, `status`). The saved file's `source` is the on-disk path, consistent with
  URL/path ingestion so it shows up in `GET /ingest/documents`.
- Same `verify_api_key` dependency and a rate limit consistent with the existing `/ingest`
  (`10/minute`).
- CORS already allows `POST`/`OPTIONS`; multipart needs no extra allowed headers.

## Pages & components

### Chat (`/`) — core
- Message thread; assistant tokens render live with a blinking cursor.
- Controls: **Agent mode** toggle (plain vs route+self-correct), **Stream** toggle (default on),
  `top_k` control (1–50, matches backend bound).
- **Agent trace**: in agent mode, `step` events render as a sequential, animated list of nodes
  (e.g. route -> retrieve -> grade -> rewrite -> generate); the final answer header shows
  `route` and `attempts`.
- **Source cards**: per source — `[n]` citation badge, `source` name, score when present, and an
  expandable snippet.
- **Usage bar**: input/output tokens, estimated cost, latency (monospace).
- **Guardrails badge**: shown when `pii_redacted` is true or `flags` is non-empty.
- Conversation is in-memory only (cleared on reload); no persistence.

### Documents (`/documents`) — the gap being closed
- **Upload**: drag-drop zone (PDF/MD) posting to `/ingest/upload`; shows per-file progress and the
  resulting chunk count; client-side guard on suffix/size mirroring the backend for fast feedback.
- **URL ingest**: text input posting to `/ingest`.
- **Document table**: `GET /ingest/documents` -> rows of source / chunks / ingested_at, each with a
  delete action (`DELETE /ingest/documents/{id}`) and an optimistic, confirmable removal.

### About (`/about`) — portfolio showcase
- Renders the architecture (the README mermaid pipeline or an equivalent SVG) plus a short prose
  walkthrough: ingest -> hybrid retrieval (RRF) -> GraphRAG expand -> Cohere rerank -> grounded,
  cited generation -> agentic self-correction. Links to the repo and the live API `/docs`.

### Shared
- **Header**: app name, nav, and a backend **status dot** driven by `GET /health/ready` (green
  ready / amber degraded / red down), with a tooltip listing dependency checks.
- **Settings panel**: API base URL and optional API key (persisted to `localStorage`).
- **Toasts**: surface `ApiError` (e.g. 429 -> "rate limited, try again shortly") and stream
  `error` events.

## Design system

- Tokens (Tailwind theme): `bg #eaebef`, `surface #ffffff`, `primary #09568c`,
  `primary-hover` (a derived darker blue), `muted #a1a8ae` (secondary text, borders, dividers),
  `ink #0f2233` (derived dark — body text & headings, for WCAG-AA contrast on the light bg),
  `danger` (one semantic red, errors only).
- Rationale for going beyond the three given colors: the supplied palette has no color dark enough
  for legible body text, and errors need red to read as errors. White surface + dark ink + one red
  are the minimum neutrals required; the three brand colors carry the identity.
- **Motion** (framer-motion, restrained): streaming token cursor, message entrance fade/slide,
  agent trace steps appearing in sequence, source-card expand/collapse, drag-over highlight on the
  upload zone, route transitions. No gratuitous animation.

## Error handling
- `apiClient` raises `ApiError(status, detail)` on non-2xx; callers show inline messages or toasts.
- 429 -> friendly rate-limit message. Network failure -> retry affordance.
- Streaming: an `error` event stops the stream and surfaces the detail; a dropped connection shows
  a retry.
- Upload: client-side suffix/size check before sending; server rejections (400/413) surfaced inline
  on the upload zone.

## Testing
- **Frontend** — Vitest + React Testing Library, with `fetch` mocked:
  - `streamNdjson` parses multi-event streams, partial lines across chunks, and `error` events.
  - Chat renders streamed tokens, source cards, usage, guardrails badge, and agent trace.
  - Documents: upload happy path + rejected suffix/size; list renders; delete removes a row.
  - `apiClient` maps non-2xx to `ApiError`.
- **Backend** — pytest for `/ingest/upload`: success (returns chunks), bad suffix (400), oversize
  (413), filename sanitization (no path traversal). Run via `./.venv/Scripts/python.exe -m pytest`.

## Deployment
- `npm run build` -> static assets, deployable to Vercel/Netlify.
- `VITE_API_URL` points at the deployed backend; backend `CORS_ORIGINS` set to the SPA origin.
- Public demo runs the backend with `API_KEY_HASH` unset (auth off); rate limiting stays on.

## Streamlit removal
- Delete `ui/streamlit_app.py` (and the empty `ui/` dir / `__pycache__`).
- Remove the `[ui]` optional-dependency group from `pyproject.toml`.
- Update `README.md`: replace the "Demo UI" section with the React app (run + build instructions),
  and drop "Streamlit" from the tech-stack line.

## Non-goals (YAGNI)
- No user accounts, multi-tenancy, or server-side conversation persistence.
- No SSR / BFF proxy (static SPA only).
- No rich document viewer/preview beyond the ingested-document table.
- No backend changes beyond `POST /ingest/upload` and the `.env.example` CORS note.

## Risks / notes
- Public auth-off demo means anyone with the URL can spend tokens; mitigated by existing rate
  limiting and by the owner controlling when the demo is live.
- Streamed tokens are raw; PII redaction/toxicity flagging applies only to the final `done` answer
  (existing backend behavior — surface this in the guardrails badge copy).
