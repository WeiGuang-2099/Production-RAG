# React Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the minimal Streamlit demo with a polished React SPA that gives every backend capability a UI entry point, backed by one new upload endpoint.

**Architecture:** A Vite + React + TypeScript static SPA in `frontend/` calls the existing FastAPI backend directly (CORS already configured). Streaming uses the backend's existing NDJSON event protocol. One backend endpoint (`POST /ingest/upload`) is added; everything else is unchanged. The old Streamlit UI is deleted.

**Tech Stack:** Backend: FastAPI, pytest. Frontend: Vite 5, React 18, TypeScript 5, Tailwind CSS 3.4, react-router-dom 6, react-markdown, framer-motion, lucide-react; tests with Vitest + React Testing Library + jsdom.

## Global Constraints

- Python tests run via `./.venv/Scripts/python.exe -m pytest` (system Python lacks deps).
- Frontend commands run inside `frontend/` (`npm install`, `npm test`, `npm run build`).
- Pin frontend deps: `vite@^5`, `react@^18`, `react-dom@^18`, `typescript@^5`, `tailwindcss@^3.4`, `react-router-dom@^6`, `vitest@^2`, `@testing-library/react@^16`, `jsdom@^25`.
- Commit messages: no Claude attribution, no co-author trailers.
- No emojis anywhere (code, copy, comments, commits).
- Palette tokens (Tailwind theme): `bg #eaebef`, `surface #ffffff`, `primary #09568c`, `primary-hover #073f66`, `muted #a1a8ae`, `ink #0f2233`, `danger #b3261e`.
- `top_k` UI bound: 1–50 (matches backend `Field(ge=1, le=50)`).
- Allowed upload suffixes: `.pdf`, `.md`, `.markdown` (mirror `app/ingestion/validation.ALLOWED_SUFFIXES`).
- The apiClient option object is `{ baseUrl: string; apiKey?: string }` everywhere.
- NDJSON stream event protocol (unchanged from backend): `step` (`node`), `sources` (`sources`), `token` (`token`), `done` (`answer`, `usage`, `latency_ms`, `guardrails`, and for agent `route`/`attempts`), `error` (`detail`).

---

### Task 1: Backend `POST /ingest/upload` endpoint

**Files:**
- Modify: `app/api/routes_ingest.py`
- Test: `tests/test_api_upload.py` (create)

**Interfaces:**
- Consumes: `ingest_pipeline(source: str, force: bool = False) -> dict`, `get_settings()`, `app.ingestion.validation.ALLOWED_SUFFIXES`.
- Produces: `POST /ingest/upload` (multipart field `file`) -> `IngestResponse` JSON `{source, chunks, status}`. 400 on bad suffix, 413 on oversize. Saves a sanitized basename under `DATA_DIR`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_upload.py`:

```python
from unittest.mock import patch


def _settings(mock_s, data_dir):
    mock_s.return_value.DATA_DIR = str(data_dir)
    mock_s.return_value.MAX_FILE_SIZE_MB = 100


def test_upload_ingests_markdown(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        _settings(mock_s, data_dir)
        mock_ingest.return_value = {"source": "x", "chunks": 3, "status": "ingested"}
        resp = client.post("/ingest/upload", files={"file": ("doc.md", b"# hi", "text/markdown")})
        assert resp.status_code == 200
        assert resp.json()["chunks"] == 3
        # file was saved under DATA_DIR and passed to the pipeline
        saved = mock_ingest.call_args[0][0]
        assert str(data_dir) in saved
        assert (data_dir / "doc.md").exists()


def test_upload_rejects_bad_suffix(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        _settings(mock_s, data_dir)
        resp = client.post("/ingest/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
        assert resp.status_code == 400
        mock_ingest.assert_not_called()


def test_upload_rejects_oversize(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_s.return_value.DATA_DIR = str(data_dir)
        mock_s.return_value.MAX_FILE_SIZE_MB = 0  # everything is oversize
        resp = client.post("/ingest/upload", files={"file": ("doc.md", b"abc", "text/markdown")})
        assert resp.status_code == 413
        mock_ingest.assert_not_called()


def test_upload_sanitizes_filename(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        _settings(mock_s, data_dir)
        mock_ingest.return_value = {"source": "x", "chunks": 1, "status": "ingested"}
        resp = client.post("/ingest/upload", files={"file": ("../../evil.md", b"x", "text/markdown")})
        assert resp.status_code == 200
        # saved as basename inside DATA_DIR, no traversal
        assert (data_dir / "evil.md").exists()
        assert not (tmp_path / "evil.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_api_upload.py -v`
Expected: FAIL (404 Not Found — endpoint does not exist yet).

- [ ] **Step 3: Implement the endpoint**

In `app/api/routes_ingest.py`, add imports near the top (keep existing imports):

```python
import os
import uuid

from fastapi import File, UploadFile

from app.ingestion.validation import ALLOWED_SUFFIXES
```

Add this route after the existing `ingest` function:

```python
@router.post("/upload", response_model=IngestResponse)
@limiter.limit("10/minute")
async def upload(request: Request, file: UploadFile = File(...), _key=Depends(verify_api_key)):
    settings = get_settings()
    name = os.path.basename(file.filename or "")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large: {size_mb:.1f}MB (max {settings.MAX_FILE_SIZE_MB}MB)")

    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / name
    if dest.exists():
        dest = data_dir / f"{Path(name).stem}-{uuid.uuid4().hex[:8]}{suffix}"
    dest.write_bytes(contents)

    result = await asyncio.to_thread(ingest_pipeline, str(dest))
    return IngestResponse(**result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_api_upload.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full backend suite (no regressions)**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/api/routes_ingest.py tests/test_api_upload.py
git commit -m "feat: POST /ingest/upload for browser file uploads"
```

---

### Task 2: CORS dev origin in `.env.example`

**Files:**
- Modify: `.env.example`

**Interfaces:**
- Produces: documented `CORS_ORIGINS` including the Vite dev origin so the SPA can call the API in development.

- [ ] **Step 1: Update `.env.example`**

Find the `CORS_ORIGINS` line in `.env.example` and set it to include the Vite dev origin, with a comment. If the line is absent, add it under a Security section:

```bash
# Comma-separated allowed origins for the browser SPA. Use "*" only for a public, auth-off demo.
# Vite dev server runs on http://localhost:5173. In production set this to your deployed SPA origin.
CORS_ORIGINS=http://localhost:5173
```

- [ ] **Step 2: Verify config still loads**

Run: `./.venv/Scripts/python.exe -c "from app.config import Settings; print(Settings(LLM_API_KEY='x', EMBEDDING_API_KEY='x', COHERE_API_KEY='x').CORS_ORIGINS)"`
Expected: prints the configured origins string without error (reads default unless `.env` overrides).

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: note Vite dev origin in CORS_ORIGINS example"
```

---

### Task 3: Remove the Streamlit demo

**Files:**
- Delete: `ui/streamlit_app.py` (and the `ui/` directory)
- Modify: `pyproject.toml` (remove `[ui]` optional-dependency group)
- Modify: `README.md` (replace the "Demo UI" section; drop "Streamlit" from tech stack)

**Interfaces:**
- Produces: a repo with a single frontend story. README "Demo UI" section temporarily points at `frontend/` (filled in fully in Task 13).

- [ ] **Step 1: Delete the Streamlit app**

```bash
git rm -r ui
```

- [ ] **Step 2: Remove the `[ui]` extra from `pyproject.toml`**

Delete these lines (the optional dependency group):

```toml
ui = [
    "streamlit>=1.40.0",
]
```

If `[project.optional-dependencies]` becomes empty, remove the empty table header too.

- [ ] **Step 3: Update README**

Replace the entire "## Demo UI" section body (currently the Streamlit instructions) with:

```markdown
## Demo UI

A React single-page app (Vite + TypeScript) in `frontend/` exposes the full system:
chat with live token streaming and cited sources, agentic mode with a visible
reasoning trace, document upload/listing/deletion, and an architecture overview.

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, talks to the API at VITE_API_URL
```
```

In the "## Tech stack" line, remove the trailing `, Streamlit`.

- [ ] **Step 4: Verify no dangling references**

Run: `git grep -in streamlit -- . ':!docs/superpowers/specs' || echo "no refs"`
Expected: `no refs` (or only the spec, which is historical).

- [ ] **Step 5: Verify backend suite still green**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove Streamlit demo in favor of React frontend"
```

---

### Task 4: Scaffold the Vite + React + TS + Tailwind app

**Files:**
- Create: `frontend/package.json`, `frontend/index.html`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/.env.example`, `frontend/.gitignore`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/index.css`, `frontend/src/vite-env.d.ts`, `frontend/src/test/setup.ts`

**Interfaces:**
- Produces: a runnable, testable shell. `npm run dev`, `npm test`, `npm run build` work. Tailwind theme exposes the palette tokens. `import.meta.env.VITE_API_URL` is the API base URL (default `http://localhost:8000`).

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "production-rag-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "framer-motion": "^11.11.0",
    "lucide-react": "^0.460.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "react-router-dom": "^6.27.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.0.1",
    "@testing-library/user-event": "^14.5.2",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "jsdom": "^25.0.1",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "typescript": "^5.6.3",
    "vite": "^5.4.10",
    "vitest": "^2.1.4"
  }
}
```

- [ ] **Step 2: Create config files**

`frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Production RAG</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`frontend/vite.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
```

`frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

`frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noEmit": true
  },
  "include": ["vite.config.ts"]
}
```

`frontend/tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#eaebef",
        surface: "#ffffff",
        primary: { DEFAULT: "#09568c", hover: "#073f66" },
        muted: "#a1a8ae",
        ink: "#0f2233",
        danger: "#b3261e",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
```

`frontend/postcss.config.js`:

```js
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

`frontend/.env.example`:

```bash
# Base URL of the running FastAPI backend.
VITE_API_URL=http://localhost:8000
```

`frontend/.gitignore`:

```
node_modules
dist
.env
```

- [ ] **Step 3: Create source shell**

`frontend/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-bg text-ink antialiased;
}
```

`frontend/src/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

`frontend/src/App.tsx`:

```tsx
export default function App() {
  return <div className="p-8 text-ink">Production RAG</div>;
}
```

`frontend/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

`frontend/src/test/setup.ts`:

```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 4: Install and verify build + a smoke test**

Add a smoke test `frontend/src/App.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders app name", () => {
  render(<App />);
  expect(screen.getByText("Production RAG")).toBeInTheDocument();
});
```

Run:
```bash
cd frontend && npm install && npm test && npm run build
```
Expected: test passes; `dist/` is produced with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -m "feat: scaffold React frontend (Vite, TS, Tailwind, Vitest)"
```

---

### Task 5: API layer — types, client, NDJSON stream parser

**Files:**
- Create: `frontend/src/api/types.ts`, `frontend/src/api/client.ts`, `frontend/src/api/stream.ts`
- Test: `frontend/src/api/client.test.ts`, `frontend/src/api/stream.test.ts`

**Interfaces:**
- Produces:
  - `types.ts`: `SourceItem`, `Usage`, `Guardrails`, `ChatResult`, `StreamEvent`, `DocumentRecord`, `IngestResult`, `HealthStatus`.
  - `client.ts`: `ApiError`, `ClientOptions = { baseUrl: string; apiKey?: string }`, `getJson<T>(o, path)`, `postJson<T>(o, path, body)`, `del<T>(o, path)`, `postStream(o, path, body): Promise<ReadableStream<Uint8Array>>`, `uploadFile(o, file): Promise<IngestResult>`.
  - `stream.ts`: `streamNdjson(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent>`.

- [ ] **Step 1: Write `types.ts`**

```ts
export interface SourceItem {
  content: string;
  metadata: { citation?: number | string; source?: string; score?: number } & Record<string, unknown>;
}
export interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  model?: string;
}
export interface Guardrails {
  pii_redacted?: string[];
  flags?: string[];
}
export interface ChatResult {
  answer: string;
  sources: SourceItem[];
  latency_ms: number;
  total_sources?: number;
  usage?: Usage;
  guardrails?: Guardrails;
  route?: string;
  attempts?: number;
}
export type StreamEvent =
  | { event: "step"; node: string }
  | { event: "sources"; sources: SourceItem[] }
  | { event: "token"; token: string }
  | {
      event: "done";
      answer: string;
      usage?: Usage;
      latency_ms?: number;
      guardrails?: Guardrails;
      route?: string;
      attempts?: number;
    }
  | { event: "error"; detail: string };
export interface DocumentRecord {
  id: string;
  source: string;
  chunks: number;
  ingested_at: string;
}
export interface IngestResult {
  source: string;
  chunks: number;
  status: string;
}
export interface HealthStatus {
  status: string;
  checks: Record<string, string>;
}
```

- [ ] **Step 2: Write the failing tests**

`frontend/src/api/stream.test.ts`:

```ts
import { streamNdjson } from "./stream";

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

test("parses newline-delimited events, including a split line across chunks", async () => {
  const body = streamFrom([
    '{"event":"sources","sources":[]}\n{"event":"to',
    'ken","token":"Hi"}\n{"event":"done","answer":"Hi"}\n',
  ]);
  const events = [];
  for await (const e of streamNdjson(body)) events.push(e);
  expect(events.map((e) => e.event)).toEqual(["sources", "token", "done"]);
});

test("yields a trailing line with no final newline", async () => {
  const body = streamFrom(['{"event":"token","token":"x"}']);
  const events = [];
  for await (const e of streamNdjson(body)) events.push(e);
  expect(events).toHaveLength(1);
});
```

`frontend/src/api/client.test.ts`:

```ts
import { afterEach, vi } from "vitest";
import { ApiError, getJson, postJson } from "./client";

const opts = { baseUrl: "http://api.test", apiKey: "secret" };

afterEach(() => vi.restoreAllMocks());

test("postJson sends auth header and returns parsed json", async () => {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", fetchMock);
  const data = await postJson<{ ok: boolean }>(opts, "/chat", { q: 1 });
  expect(data.ok).toBe(true);
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toBe("http://api.test/chat");
  expect((init.headers as Record<string, string>).Authorization).toBe("Bearer secret");
});

test("non-2xx raises ApiError with detail", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "rate limited" }), { status: 429 })),
  );
  await expect(getJson(opts, "/x")).rejects.toMatchObject({ status: 429, detail: "rate limited" });
  await expect(getJson(opts, "/x")).rejects.toBeInstanceOf(ApiError);
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/api`
Expected: FAIL (modules `./stream`, `./client` not found).

- [ ] **Step 4: Implement `stream.ts`**

```ts
import type { StreamEvent } from "./types";

export async function* streamNdjson(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line) as StreamEvent;
    }
  }
  const rest = buf.trim();
  if (rest) yield JSON.parse(rest) as StreamEvent;
}
```

- [ ] **Step 5: Implement `client.ts`**

```ts
import type { IngestResult } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(detail);
    this.name = "ApiError";
  }
}

export interface ClientOptions {
  baseUrl: string;
  apiKey?: string;
}

function authHeaders(o: ClientOptions, extra: Record<string, string> = {}): Record<string, string> {
  return o.apiKey ? { ...extra, Authorization: `Bearer ${o.apiKey}` } : { ...extra };
}

async function detailOf(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (data?.detail) return JSON.stringify(data.detail);
    return JSON.stringify(data);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function getJson<T>(o: ClientOptions, path: string): Promise<T> {
  const res = await fetch(o.baseUrl + path, { headers: authHeaders(o) });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function postJson<T>(o: ClientOptions, path: string, body: unknown): Promise<T> {
  const res = await fetch(o.baseUrl + path, {
    method: "POST",
    headers: authHeaders(o, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function del<T>(o: ClientOptions, path: string): Promise<T> {
  const res = await fetch(o.baseUrl + path, { method: "DELETE", headers: authHeaders(o) });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function postStream(o: ClientOptions, path: string, body: unknown): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(o.baseUrl + path, {
    method: "POST",
    headers: authHeaders(o, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  if (!res.body) throw new ApiError(0, "empty response body");
  return res.body;
}

export async function uploadFile(o: ClientOptions, file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(o.baseUrl + "/ingest/upload", { method: "POST", headers: authHeaders(o), body: form });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as IngestResult;
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/api`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api
git commit -m "feat: frontend API client, types, and NDJSON stream parser"
```

---

### Task 6: Settings + Toast context

**Files:**
- Create: `frontend/src/context/SettingsContext.tsx`, `frontend/src/context/ToastContext.tsx`
- Test: `frontend/src/context/SettingsContext.test.tsx`

**Interfaces:**
- Produces:
  - `SettingsProvider`, `useSettings(): { client: ClientOptions; setBaseUrl(s: string): void; setApiKey(s: string): void }`. Persists `baseUrl`/`apiKey` to `localStorage`; `baseUrl` defaults to `import.meta.env.VITE_API_URL ?? "http://localhost:8000"`.
  - `ToastProvider`, `useToast(): { toast(message: string, kind?: "info" | "error"): void }`, and a `<Toaster />` rendering current toasts.

- [ ] **Step 1: Write the failing test**

`frontend/src/context/SettingsContext.test.tsx`:

```tsx
import { act, renderHook } from "@testing-library/react";
import { SettingsProvider, useSettings } from "./SettingsContext";

test("defaults baseUrl and persists overrides", () => {
  localStorage.clear();
  const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>;
  const { result } = renderHook(() => useSettings(), { wrapper });
  expect(result.current.client.baseUrl).toBe("http://localhost:8000");
  act(() => result.current.setApiKey("k"));
  expect(result.current.client.apiKey).toBe("k");
  expect(localStorage.getItem("rag.apiKey")).toBe("k");
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/context`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `SettingsContext.tsx`**

```tsx
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ClientOptions } from "../api/client";

interface SettingsValue {
  client: ClientOptions;
  setBaseUrl: (s: string) => void;
  setApiKey: (s: string) => void;
}

const DEFAULT_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const SettingsContext = createContext<SettingsValue | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [baseUrl, setBaseUrlState] = useState(() => localStorage.getItem("rag.baseUrl") ?? DEFAULT_BASE);
  const [apiKey, setApiKeyState] = useState(() => localStorage.getItem("rag.apiKey") ?? "");

  const setBaseUrl = useCallback((s: string) => {
    setBaseUrlState(s);
    localStorage.setItem("rag.baseUrl", s);
  }, []);
  const setApiKey = useCallback((s: string) => {
    setApiKeyState(s);
    localStorage.setItem("rag.apiKey", s);
  }, []);

  const value = useMemo<SettingsValue>(
    () => ({ client: { baseUrl, apiKey: apiKey || undefined }, setBaseUrl, setApiKey }),
    [baseUrl, apiKey, setBaseUrl, setApiKey],
  );
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): SettingsValue {
  const v = useContext(SettingsContext);
  if (!v) throw new Error("useSettings must be used within SettingsProvider");
  return v;
}
```

Note: import is `useCallback` (lowercase c). Use:

```tsx
import { createContext, useCallback, useContext, useMemo, useState } from "react";
```

- [ ] **Step 4: Implement `ToastContext.tsx`**

```tsx
import { AnimatePresence, motion } from "framer-motion";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

interface Toast {
  id: number;
  message: string;
  kind: "info" | "error";
}
interface ToastValue {
  toast: (message: string, kind?: "info" | "error") => void;
}

const ToastContext = createContext<ToastValue | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toast = useCallback((message: string, kind: "info" | "error" = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, message, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  }, []);
  const value = useMemo(() => ({ toast }), [toast]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 12 }}
              className={`rounded-md px-4 py-2 text-sm text-white shadow-lg ${
                t.kind === "error" ? "bg-danger" : "bg-primary"
              }`}
            >
              {t.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastValue {
  const v = useContext(ToastContext);
  if (!v) throw new Error("useToast must be used within ToastProvider");
  return v;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/context`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/context
git commit -m "feat: settings and toast contexts"
```

---

### Task 7: App shell — router, header, health status, settings panel

**Files:**
- Create: `frontend/src/hooks/useHealth.ts`, `frontend/src/components/StatusDot.tsx`, `frontend/src/components/SettingsPanel.tsx`, `frontend/src/components/Header.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/main.tsx`
- Test: `frontend/src/hooks/useHealth.test.ts`

**Interfaces:**
- Consumes: `useSettings`, `getJson`, `HealthStatus`.
- Produces:
  - `useHealth(): { status: "loading" | "ready" | "degraded" | "down"; checks: Record<string,string> }` — polls `GET /health/ready` every 15s.
  - `Header` with nav links (`/`, `/documents`, `/about`), a `StatusDot`, and a settings toggle.
  - `App` renders `<BrowserRouter>` with routes for the three pages (placeholder page bodies until Tasks 10-12).

- [ ] **Step 1: Write the failing test for `useHealth`**

`frontend/src/hooks/useHealth.test.ts`:

```ts
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { useHealth } from "./useHealth";

afterEach(() => vi.restoreAllMocks());

const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>;

test("maps a ready response to ready", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "ready", checks: { app: "ok", qdrant: "ok" } }), { status: 200 }),
    ),
  );
  const { result } = renderHook(() => useHealth(), { wrapper });
  await waitFor(() => expect(result.current.status).toBe("ready"));
});

test("maps a failed fetch to down", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network")));
  const { result } = renderHook(() => useHealth(), { wrapper });
  await waitFor(() => expect(result.current.status).toBe("down"));
});
```

Note: this test file uses JSX in the wrapper, so name it `useHealth.test.tsx` instead of `.ts`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/hooks`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `useHealth.ts`**

```ts
import { useEffect, useState } from "react";
import { getJson } from "../api/client";
import type { HealthStatus } from "../api/types";
import { useSettings } from "../context/SettingsContext";

type Status = "loading" | "ready" | "degraded" | "down";

export function useHealth(): { status: Status; checks: Record<string, string> } {
  const { client } = useSettings();
  const [status, setStatus] = useState<Status>("loading");
  const [checks, setChecks] = useState<Record<string, string>>({});

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const data = await getJson<HealthStatus>(client, "/health/ready");
        if (!active) return;
        setChecks(data.checks ?? {});
        const anyFailed = Object.values(data.checks ?? {}).some((v) => v.includes("failed") || v === "empty");
        setStatus(data.status === "ready" ? (anyFailed ? "degraded" : "ready") : "degraded");
      } catch {
        if (active) setStatus("down");
      }
    }
    poll();
    const id = setInterval(poll, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [client]);

  return { status, checks };
}
```

- [ ] **Step 4: Implement `StatusDot.tsx`**

```tsx
import { useHealth } from "../hooks/useHealth";

const COLOR: Record<string, string> = {
  loading: "bg-muted",
  ready: "bg-emerald-500",
  degraded: "bg-amber-500",
  down: "bg-danger",
};

export function StatusDot() {
  const { status, checks } = useHealth();
  const tip = Object.entries(checks)
    .map(([k, v]) => `${k}: ${v}`)
    .join("\n");
  return (
    <span className="flex items-center gap-2 text-xs text-muted" title={tip || status}>
      <span className={`h-2.5 w-2.5 rounded-full ${COLOR[status]}`} />
      {status}
    </span>
  );
}
```

- [ ] **Step 5: Implement `SettingsPanel.tsx`**

```tsx
import { motion } from "framer-motion";
import { useSettings } from "../context/SettingsContext";

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const { client, setBaseUrl, setApiKey } = useSettings();
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="absolute right-0 top-12 z-40 w-80 rounded-lg border border-muted/40 bg-surface p-4 shadow-xl"
    >
      <label className="block text-xs font-medium text-muted">API URL</label>
      <input
        className="mt-1 w-full rounded border border-muted/50 px-2 py-1 text-sm"
        value={client.baseUrl}
        onChange={(e) => setBaseUrl(e.target.value)}
      />
      <label className="mt-3 block text-xs font-medium text-muted">API key (optional)</label>
      <input
        type="password"
        className="mt-1 w-full rounded border border-muted/50 px-2 py-1 text-sm"
        value={client.apiKey ?? ""}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <button className="mt-4 w-full rounded bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary-hover" onClick={onClose}>
        Done
      </button>
    </motion.div>
  );
}
```

- [ ] **Step 6: Implement `Header.tsx`**

```tsx
import { Settings as SettingsIcon } from "lucide-react";
import { useState } from "react";
import { NavLink } from "react-router-dom";
import { SettingsPanel } from "./SettingsPanel";
import { StatusDot } from "./StatusDot";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded text-sm ${isActive ? "bg-primary text-white" : "text-ink hover:bg-muted/20"}`;

export function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="relative flex items-center justify-between border-b border-muted/30 bg-surface px-6 py-3">
      <div className="flex items-center gap-6">
        <span className="font-semibold text-primary">Production RAG</span>
        <nav className="flex gap-1">
          <NavLink to="/" end className={linkClass}>
            Chat
          </NavLink>
          <NavLink to="/documents" className={linkClass}>
            Documents
          </NavLink>
          <NavLink to="/about" className={linkClass}>
            About
          </NavLink>
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <StatusDot />
        <button aria-label="Settings" className="text-muted hover:text-ink" onClick={() => setOpen((o) => !o)}>
          <SettingsIcon size={18} />
        </button>
        {open && <SettingsPanel onClose={() => setOpen(false)} />}
      </div>
    </header>
  );
}
```

- [ ] **Step 7: Wire `App.tsx` and `main.tsx`**

`frontend/src/App.tsx`:

```tsx
import { Route, Routes } from "react-router-dom";
import { Header } from "./components/Header";
import { AboutPage } from "./pages/AboutPage";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";

export default function App() {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  );
}
```

Create placeholder pages so this compiles now (replaced in later tasks):

`frontend/src/pages/ChatPage.tsx`:

```tsx
export function ChatPage() {
  return <div>Chat</div>;
}
```

`frontend/src/pages/DocumentsPage.tsx`:

```tsx
export function DocumentsPage() {
  return <div>Documents</div>;
}
```

`frontend/src/pages/AboutPage.tsx`:

```tsx
export function AboutPage() {
  return <div>About</div>;
}
```

`frontend/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { SettingsProvider } from "./context/SettingsContext";
import { ToastProvider } from "./context/ToastContext";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <SettingsProvider>
        <ToastProvider>
          <App />
        </ToastProvider>
      </SettingsProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
```

Delete the now-obsolete `frontend/src/App.test.tsx` (the smoke test asserted the old body):

```bash
git rm frontend/src/App.test.tsx
```

- [ ] **Step 8: Run tests and typecheck**

Run: `cd frontend && npx vitest run && npm run build`
Expected: health tests pass; build succeeds.

- [ ] **Step 9: Commit**

```bash
git add frontend
git commit -m "feat: app shell with router, header, health status, settings"
```

---

### Task 8: `useChat` hook

**Files:**
- Create: `frontend/src/hooks/useChat.ts`
- Test: `frontend/src/hooks/useChat.test.tsx`

**Interfaces:**
- Consumes: `useSettings`, `postJson`, `postStream`, `streamNdjson`, `ChatResult`, `StreamEvent`, `SourceItem`.
- Produces:
  - Types `ChatMessage = { role: "user" | "assistant"; content: string; sources?: SourceItem[]; usage?: Usage; guardrails?: Guardrails; latency_ms?: number; route?: string; attempts?: number; steps?: string[] }`.
  - `useChat(): { messages: ChatMessage[]; busy: boolean; send(question: string, opts: { agent: boolean; stream: boolean; topK: number }): Promise<void> }`.

- [ ] **Step 1: Write the failing test**

`frontend/src/hooks/useChat.test.tsx`:

```tsx
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { ToastProvider } from "../context/ToastContext";
import { useChat } from "./useChat";

afterEach(() => vi.restoreAllMocks());

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <SettingsProvider>
    <ToastProvider>{children}</ToastProvider>
  </SettingsProvider>
);

function ndjsonResponse(lines: string[]): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream({
    start(c) {
      for (const l of lines) c.enqueue(enc.encode(l + "\n"));
      c.close();
    },
  });
  return new Response(body, { status: 200 });
}

test("streaming appends tokens and finalizes from the done event", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      ndjsonResponse([
        '{"event":"sources","sources":[{"content":"c","metadata":{"citation":1}}]}',
        '{"event":"token","token":"Hel"}',
        '{"event":"token","token":"lo"}',
        '{"event":"done","answer":"Hello","usage":{"output_tokens":2},"latency_ms":12}',
      ]),
    ),
  );
  const { result } = renderHook(() => useChat(), { wrapper });
  await act(async () => {
    await result.current.send("hi", { agent: false, stream: true, topK: 5 });
  });
  await waitFor(() => expect(result.current.busy).toBe(false));
  const last = result.current.messages.at(-1)!;
  expect(last.role).toBe("assistant");
  expect(last.content).toBe("Hello");
  expect(last.sources).toHaveLength(1);
  expect(last.usage?.output_tokens).toBe(2);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/hooks/useChat.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `useChat.ts`**

```ts
import { useCallback, useState } from "react";
import { ApiError, postJson, postStream } from "../api/client";
import { streamNdjson } from "../api/stream";
import type { ChatResult, Guardrails, SourceItem, Usage } from "../api/types";
import { useSettings } from "../context/SettingsContext";
import { useToast } from "../context/ToastContext";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  usage?: Usage;
  guardrails?: Guardrails;
  latency_ms?: number;
  route?: string;
  attempts?: number;
  steps?: string[];
}

export interface SendOpts {
  agent: boolean;
  stream: boolean;
  topK: number;
}

export function useChat() {
  const { client } = useSettings();
  const { toast } = useToast();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);

  const patchLast = useCallback((patch: Partial<ChatMessage> | ((m: ChatMessage) => ChatMessage)) => {
    setMessages((ms) => {
      const copy = ms.slice();
      const i = copy.length - 1;
      copy[i] = typeof patch === "function" ? patch(copy[i]) : { ...copy[i], ...patch };
      return copy;
    });
  }, []);

  const send = useCallback(
    async (question: string, opts: SendOpts) => {
      if (!question.trim() || busy) return;
      setBusy(true);
      setMessages((ms) => [...ms, { role: "user", content: question }, { role: "assistant", content: "" }]);
      const base = opts.agent ? "/agent" : "/chat";
      const body = { question, top_k: opts.topK };
      try {
        if (opts.stream) {
          const stream = await postStream(client, `${base}/stream`, body);
          for await (const ev of streamNdjson(stream)) {
            if (ev.event === "step") patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), ev.node] }));
            else if (ev.event === "sources") patchLast({ sources: ev.sources });
            else if (ev.event === "token") patchLast((m) => ({ ...m, content: m.content + ev.token }));
            else if (ev.event === "done")
              patchLast((m) => ({
                ...m,
                content: ev.answer ?? m.content,
                usage: ev.usage,
                guardrails: ev.guardrails,
                latency_ms: ev.latency_ms,
                route: ev.route,
                attempts: ev.attempts,
              }));
            else if (ev.event === "error") {
              toast(ev.detail, "error");
              patchLast({ content: `Error: ${ev.detail}` });
            }
          }
        } else {
          const r = await postJson<ChatResult>(client, base, body);
          patchLast({
            content: r.answer,
            sources: r.sources,
            usage: r.usage,
            guardrails: r.guardrails,
            latency_ms: r.latency_ms,
            route: r.route,
            attempts: r.attempts,
          });
        }
      } catch (e) {
        const msg = e instanceof ApiError ? `${e.status}: ${e.detail}` : "Request failed";
        toast(msg, "error");
        patchLast({ content: `Error: ${msg}` });
      } finally {
        setBusy(false);
      }
    },
    [busy, client, patchLast, toast],
  );

  return { messages, busy, send };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run src/hooks/useChat.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useChat.ts frontend/src/hooks/useChat.test.tsx
git commit -m "feat: useChat hook (streaming + non-streaming, chat + agent)"
```

---

### Task 9: Chat UI components

**Files:**
- Create: `frontend/src/components/chat/SourceCards.tsx`, `UsageBar.tsx`, `GuardrailsBadge.tsx`, `AgentTrace.tsx`, `ChatControls.tsx`, `MessageThread.tsx`
- Test: `frontend/src/components/chat/SourceCards.test.tsx`, `UsageBar.test.tsx`

**Interfaces:**
- Consumes: `SourceItem`, `Usage`, `Guardrails`, `ChatMessage`.
- Produces presentational components:
  - `SourceCards({ sources: SourceItem[] })`
  - `UsageBar({ usage?: Usage; latency_ms?: number })`
  - `GuardrailsBadge({ guardrails?: Guardrails })`
  - `AgentTrace({ steps?: string[]; route?: string; attempts?: number })`
  - `ChatControls({ agent, stream, topK, busy, onAgent, onStream, onTopK, onSend })`
  - `MessageThread({ messages: ChatMessage[] })`

- [ ] **Step 1: Write failing tests**

`frontend/src/components/chat/UsageBar.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { UsageBar } from "./UsageBar";

test("renders tokens, cost, latency", () => {
  render(<UsageBar usage={{ input_tokens: 50, output_tokens: 10, cost_usd: 0.0002 }} latency_ms={123} />);
  expect(screen.getByText(/50/)).toBeInTheDocument();
  expect(screen.getByText(/\$0.00020/)).toBeInTheDocument();
  expect(screen.getByText(/123 ms/)).toBeInTheDocument();
});
```

`frontend/src/components/chat/SourceCards.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { SourceCards } from "./SourceCards";

test("renders a citation badge and source name", () => {
  render(
    <SourceCards sources={[{ content: "snippet text", metadata: { citation: 1, source: "doc.pdf", score: 0.82 } }]} />,
  );
  expect(screen.getByText("[1]")).toBeInTheDocument();
  expect(screen.getByText(/doc\.pdf/)).toBeInTheDocument();
});

test("renders nothing for empty sources", () => {
  const { container } = render(<SourceCards sources={[]} />);
  expect(container).toBeEmptyDOMElement();
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run src/components/chat`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement the presentational components**

`frontend/src/components/chat/UsageBar.tsx`:

```tsx
import type { Usage } from "../../api/types";

export function UsageBar({ usage, latency_ms }: { usage?: Usage; latency_ms?: number }) {
  if (!usage && latency_ms == null) return null;
  const cell = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-muted">{label}</span>
      <span className="font-mono text-sm text-ink">{value}</span>
    </div>
  );
  const cost = usage?.cost_usd;
  return (
    <div className="mt-2 flex gap-6 rounded-md bg-bg px-3 py-2">
      {cell("in", String(usage?.input_tokens ?? "-"))}
      {cell("out", String(usage?.output_tokens ?? "-"))}
      {cell("cost", typeof cost === "number" ? `$${cost.toFixed(5)}` : "-")}
      {cell("latency", latency_ms != null ? `${Math.round(latency_ms)} ms` : "-")}
    </div>
  );
}
```

`frontend/src/components/chat/SourceCards.tsx`:

```tsx
import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type { SourceItem } from "../../api/types";

export function SourceCards({ sources }: { sources: SourceItem[] }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-3">
      <button className="text-sm text-primary hover:underline" onClick={() => setOpen((o) => !o)}>
        Sources ({sources.length})
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 space-y-2 overflow-hidden"
          >
            {sources.map((s, i) => {
              const citation = s.metadata?.citation ?? i + 1;
              const score = s.metadata?.score;
              return (
                <li key={i} className="rounded-md border border-muted/30 bg-surface p-3 text-sm">
                  <div className="flex items-center gap-2 font-mono text-xs text-muted">
                    <span className="text-primary">[{String(citation)}]</span>
                    <span className="truncate">{s.metadata?.source ?? "unknown"}</span>
                    {typeof score === "number" && <span>· {score.toFixed(2)}</span>}
                  </div>
                  <p className="mt-1 text-ink/80">{s.content.slice(0, 500)}</p>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
```

`frontend/src/components/chat/GuardrailsBadge.tsx`:

```tsx
import { ShieldCheck } from "lucide-react";
import type { Guardrails } from "../../api/types";

export function GuardrailsBadge({ guardrails }: { guardrails?: Guardrails }) {
  const redacted = guardrails?.pii_redacted ?? [];
  const flags = guardrails?.flags ?? [];
  if (redacted.length === 0 && flags.length === 0) return null;
  const parts = [...redacted.map((r) => `PII:${r}`), ...flags.map((f) => `flag:${f}`)];
  return (
    <span className="mt-2 inline-flex items-center gap-1 rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-800">
      <ShieldCheck size={12} />
      {parts.join(", ")}
    </span>
  );
}
```

`frontend/src/components/chat/AgentTrace.tsx`:

```tsx
import { motion } from "framer-motion";

export function AgentTrace({ steps, route, attempts }: { steps?: string[]; route?: string; attempts?: number }) {
  if ((!steps || steps.length === 0) && !route) return null;
  return (
    <div className="mb-2 rounded-md border border-muted/30 bg-bg px-3 py-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {route && <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-primary">route: {route}</span>}
        {typeof attempts === "number" && <span className="text-muted">attempts: {attempts}</span>}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {(steps ?? []).map((s, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            className="rounded bg-surface px-2 py-0.5 font-mono text-[11px] text-ink/70"
          >
            {s}
          </motion.span>
        ))}
      </div>
    </div>
  );
}
```

`frontend/src/components/chat/ChatControls.tsx`:

```tsx
import { Send } from "lucide-react";
import { useState } from "react";

interface Props {
  agent: boolean;
  stream: boolean;
  topK: number;
  busy: boolean;
  onAgent: (v: boolean) => void;
  onStream: (v: boolean) => void;
  onTopK: (v: number) => void;
  onSend: (q: string) => void;
}

export function ChatControls({ agent, stream, topK, busy, onAgent, onStream, onTopK, onSend }: Props) {
  const [text, setText] = useState("");
  const submit = () => {
    if (!text.trim()) return;
    onSend(text);
    setText("");
  };
  return (
    <div className="border-t border-muted/30 bg-surface p-3">
      <div className="mb-2 flex flex-wrap items-center gap-4 text-sm text-muted">
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={agent} onChange={(e) => onAgent(e.target.checked)} /> Agent mode
        </label>
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={stream} onChange={(e) => onStream(e.target.checked)} /> Stream
        </label>
        <label className="flex items-center gap-1">
          top_k
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => onTopK(Math.min(50, Math.max(1, Number(e.target.value) || 1)))}
            className="w-16 rounded border border-muted/50 px-1"
          />
        </label>
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 rounded-md border border-muted/50 px-3 py-2"
          placeholder="Ask a question about the ingested documents"
          value={text}
          disabled={busy}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        <button
          className="flex items-center gap-1 rounded-md bg-primary px-4 py-2 text-white hover:bg-primary-hover disabled:opacity-50"
          disabled={busy}
          onClick={submit}
        >
          <Send size={16} /> Send
        </button>
      </div>
    </div>
  );
}
```

`frontend/src/components/chat/MessageThread.tsx`:

```tsx
import { motion } from "framer-motion";
import Markdown from "react-markdown";
import type { ChatMessage } from "../../hooks/useChat";
import { AgentTrace } from "./AgentTrace";
import { GuardrailsBadge } from "./GuardrailsBadge";
import { SourceCards } from "./SourceCards";
import { UsageBar } from "./UsageBar";

export function MessageThread({ messages }: { messages: ChatMessage[] }) {
  return (
    <div className="space-y-4">
      {messages.map((m, i) => (
        <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
          <div className={`text-xs font-medium ${m.role === "user" ? "text-primary" : "text-muted"}`}>{m.role}</div>
          {m.role === "assistant" && <AgentTrace steps={m.steps} route={m.route} attempts={m.attempts} />}
          <div className="prose prose-sm max-w-none text-ink">
            <Markdown>{m.content || "..."}</Markdown>
          </div>
          {m.role === "assistant" && (
            <>
              <GuardrailsBadge guardrails={m.guardrails} />
              <SourceCards sources={m.sources ?? []} />
              <UsageBar usage={m.usage} latency_ms={m.latency_ms} />
            </>
          )}
        </motion.div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/chat`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/chat
git commit -m "feat: chat UI components (sources, usage, guardrails, agent trace)"
```

---

### Task 10: Chat page

**Files:**
- Modify: `frontend/src/pages/ChatPage.tsx`
- Test: `frontend/src/pages/ChatPage.test.tsx`

**Interfaces:**
- Consumes: `useChat`, `MessageThread`, `ChatControls`.
- Produces: the wired Chat page (state for `agent`/`stream`/`topK`, default `stream=true`, `topK=5`).

- [ ] **Step 1: Write the failing test**

`frontend/src/pages/ChatPage.test.tsx`:

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { ToastProvider } from "../context/ToastContext";
import { ChatPage } from "./ChatPage";

afterEach(() => vi.restoreAllMocks());

function ui() {
  return render(
    <SettingsProvider>
      <ToastProvider>
        <ChatPage />
      </ToastProvider>
    </SettingsProvider>,
  );
}

test("submitting a question renders the streamed answer", async () => {
  const enc = new TextEncoder();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(enc.encode('{"event":"token","token":"Hi there"}\n'));
            c.enqueue(enc.encode('{"event":"done","answer":"Hi there"}\n'));
            c.close();
          },
        }),
        { status: 200 },
      ),
    ),
  );
  ui();
  await userEvent.type(screen.getByPlaceholderText(/Ask a question/i), "hello");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));
  await waitFor(() => expect(screen.getByText("Hi there")).toBeInTheDocument());
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/pages/ChatPage.test.tsx`
Expected: FAIL (placeholder page has no input).

- [ ] **Step 3: Implement `ChatPage.tsx`**

```tsx
import { useState } from "react";
import { ChatControls } from "../components/chat/ChatControls";
import { MessageThread } from "../components/chat/MessageThread";
import { useChat } from "../hooks/useChat";

export function ChatPage() {
  const { messages, busy, send } = useChat();
  const [agent, setAgent] = useState(false);
  const [stream, setStream] = useState(true);
  const [topK, setTopK] = useState(5);

  return (
    <div className="flex h-[calc(100vh-9rem)] flex-col rounded-lg border border-muted/30 bg-surface">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted">Ask a question about your ingested documents.</p>
        ) : (
          <MessageThread messages={messages} />
        )}
      </div>
      <ChatControls
        agent={agent}
        stream={stream}
        topK={topK}
        busy={busy}
        onAgent={setAgent}
        onStream={setStream}
        onTopK={setTopK}
        onSend={(q) => send(q, { agent, stream, topK })}
      />
    </div>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run src/pages/ChatPage.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/ChatPage.tsx frontend/src/pages/ChatPage.test.tsx
git commit -m "feat: chat page"
```

---

### Task 11: Documents page (upload, URL ingest, list, delete)

**Files:**
- Create: `frontend/src/hooks/useDocuments.ts`, `frontend/src/components/documents/UploadZone.tsx`, `UrlIngest.tsx`, `DocumentTable.tsx`
- Modify: `frontend/src/pages/DocumentsPage.tsx`
- Test: `frontend/src/hooks/useDocuments.test.tsx`, `frontend/src/components/documents/UploadZone.test.tsx`

**Interfaces:**
- Consumes: `useSettings`, `getJson`, `del`, `postJson`, `uploadFile`, `DocumentRecord`, `IngestResult`.
- Produces:
  - `useDocuments(): { docs: DocumentRecord[]; loading: boolean; refresh(): Promise<void>; remove(id: string): Promise<void>; ingestUrl(url: string): Promise<void>; ingestFile(file: File): Promise<void> }`.
  - `UploadZone({ onFile, accept })`, `UrlIngest({ onSubmit })`, `DocumentTable({ docs, onDelete })`.

- [ ] **Step 1: Write failing tests**

`frontend/src/hooks/useDocuments.test.tsx`:

```tsx
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { ToastProvider } from "../context/ToastContext";
import { useDocuments } from "./useDocuments";

afterEach(() => vi.restoreAllMocks());

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <SettingsProvider>
    <ToastProvider>{children}</ToastProvider>
  </SettingsProvider>
);

test("refresh loads documents", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify([{ id: "a", source: "doc.pdf", chunks: 3, ingested_at: "2026-06-22" }]), {
        status: 200,
      }),
    ),
  );
  const { result } = renderHook(() => useDocuments(), { wrapper });
  await act(async () => result.current.refresh());
  await waitFor(() => expect(result.current.docs).toHaveLength(1));
  expect(result.current.docs[0].source).toBe("doc.pdf");
});
```

`frontend/src/components/documents/UploadZone.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { UploadZone } from "./UploadZone";

test("rejects an unsupported file type without calling onFile", async () => {
  const onFile = vi.fn();
  render(<UploadZone onFile={onFile} />);
  const input = screen.getByTestId("file-input") as HTMLInputElement;
  await userEvent.upload(input, new File(["x"], "evil.exe", { type: "application/octet-stream" }));
  expect(onFile).not.toHaveBeenCalled();
  expect(screen.getByText(/unsupported/i)).toBeInTheDocument();
});

test("accepts a markdown file", async () => {
  const onFile = vi.fn();
  render(<UploadZone onFile={onFile} />);
  const input = screen.getByTestId("file-input") as HTMLInputElement;
  await userEvent.upload(input, new File(["# hi"], "doc.md", { type: "text/markdown" }));
  expect(onFile).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run src/hooks/useDocuments.test.tsx src/components/documents`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement `useDocuments.ts`**

```ts
import { useCallback, useState } from "react";
import { ApiError, del, getJson, postJson, uploadFile } from "../api/client";
import type { DocumentRecord, IngestResult } from "../api/types";
import { useSettings } from "../context/SettingsContext";
import { useToast } from "../context/ToastContext";

export function useDocuments() {
  const { client } = useSettings();
  const { toast } = useToast();
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(false);

  const fail = useCallback(
    (e: unknown) => toast(e instanceof ApiError ? `${e.status}: ${e.detail}` : "Request failed", "error"),
    [toast],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setDocs(await getJson<DocumentRecord[]>(client, "/ingest/documents"));
    } catch (e) {
      fail(e);
    } finally {
      setLoading(false);
    }
  }, [client, fail]);

  const remove = useCallback(
    async (id: string) => {
      try {
        await del(client, `/ingest/documents/${id}`);
        setDocs((d) => d.filter((x) => x.id !== id));
        toast("Document removed");
      } catch (e) {
        fail(e);
      }
    },
    [client, fail, toast],
  );

  const ingestUrl = useCallback(
    async (url: string) => {
      try {
        const r = await postJson<IngestResult>(client, "/ingest", { source: url });
        toast(`Ingested ${r.chunks} chunks`);
        await refresh();
      } catch (e) {
        fail(e);
      }
    },
    [client, fail, refresh, toast],
  );

  const ingestFile = useCallback(
    async (file: File) => {
      try {
        const r = await uploadFile(client, file);
        toast(`Ingested ${r.chunks} chunks`);
        await refresh();
      } catch (e) {
        fail(e);
      }
    },
    [client, fail, refresh, toast],
  );

  return { docs, loading, refresh, remove, ingestUrl, ingestFile };
}
```

- [ ] **Step 4: Implement the components**

`frontend/src/components/documents/UploadZone.tsx`:

```tsx
import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

const ALLOWED = [".pdf", ".md", ".markdown"];

function allowed(name: string): boolean {
  return ALLOWED.some((s) => name.toLowerCase().endsWith(s));
}

export function UploadZone({ onFile }: { onFile: (f: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");

  const handle = (file: File | undefined) => {
    if (!file) return;
    if (!allowed(file.name)) {
      setError(`Unsupported file type. Allowed: ${ALLOWED.join(", ")}`);
      return;
    }
    setError("");
    onFile(file);
  };

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          handle(e.dataTransfer.files[0]);
        }}
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-8 text-sm transition ${
          over ? "border-primary bg-primary/5" : "border-muted/50 text-muted"
        }`}
      >
        <UploadCloud size={28} />
        Drop a PDF or Markdown file here, or click to choose
        <input
          ref={inputRef}
          data-testid="file-input"
          type="file"
          accept={ALLOWED.join(",")}
          className="hidden"
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>
      {error && <p className="mt-2 text-sm text-danger">{error}</p>}
    </div>
  );
}
```

`frontend/src/components/documents/UrlIngest.tsx`:

```tsx
import { useState } from "react";

export function UrlIngest({ onSubmit }: { onSubmit: (url: string) => void }) {
  const [url, setUrl] = useState("");
  return (
    <div className="flex gap-2">
      <input
        className="flex-1 rounded-md border border-muted/50 px-3 py-2 text-sm"
        placeholder="https://example.com/document.pdf"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <button
        className="rounded-md bg-primary px-4 py-2 text-sm text-white hover:bg-primary-hover disabled:opacity-50"
        disabled={!url.trim()}
        onClick={() => {
          onSubmit(url.trim());
          setUrl("");
        }}
      >
        Ingest URL
      </button>
    </div>
  );
}
```

`frontend/src/components/documents/DocumentTable.tsx`:

```tsx
import { Trash2 } from "lucide-react";
import type { DocumentRecord } from "../../api/types";

export function DocumentTable({ docs, onDelete }: { docs: DocumentRecord[]; onDelete: (id: string) => void }) {
  if (docs.length === 0) return <p className="text-sm text-muted">No documents ingested yet.</p>;
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-xs uppercase text-muted">
        <tr>
          <th className="py-2">Source</th>
          <th className="py-2">Chunks</th>
          <th className="py-2">Ingested</th>
          <th className="py-2" />
        </tr>
      </thead>
      <tbody>
        {docs.map((d) => (
          <tr key={d.id} className="border-t border-muted/20">
            <td className="max-w-xs truncate py-2 font-mono text-xs">{d.source}</td>
            <td className="py-2">{d.chunks}</td>
            <td className="py-2 text-muted">{d.ingested_at}</td>
            <td className="py-2 text-right">
              <button aria-label={`Delete ${d.source}`} className="text-muted hover:text-danger" onClick={() => onDelete(d.id)}>
                <Trash2 size={16} />
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 5: Implement `DocumentsPage.tsx`**

```tsx
import { useEffect } from "react";
import { DocumentTable } from "../components/documents/DocumentTable";
import { UploadZone } from "../components/documents/UploadZone";
import { UrlIngest } from "../components/documents/UrlIngest";
import { useDocuments } from "../hooks/useDocuments";

export function DocumentsPage() {
  const { docs, refresh, remove, ingestUrl, ingestFile } = useDocuments();
  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-muted/30 bg-surface p-4">
        <h2 className="mb-3 text-sm font-semibold text-ink">Add documents</h2>
        <UploadZone onFile={ingestFile} />
        <div className="mt-4">
          <UrlIngest onSubmit={ingestUrl} />
        </div>
      </section>
      <section className="rounded-lg border border-muted/30 bg-surface p-4">
        <h2 className="mb-3 text-sm font-semibold text-ink">Ingested documents</h2>
        <DocumentTable docs={docs} onDelete={remove} />
      </section>
    </div>
  );
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/hooks/useDocuments.test.tsx src/components/documents`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/useDocuments.ts frontend/src/hooks/useDocuments.test.tsx frontend/src/components/documents frontend/src/pages/DocumentsPage.tsx
git commit -m "feat: documents page (upload, URL ingest, list, delete)"
```

---

### Task 12: About page

**Files:**
- Modify: `frontend/src/pages/AboutPage.tsx`

**Interfaces:**
- Produces: a static architecture overview page. No backend calls.

- [ ] **Step 1: Implement `AboutPage.tsx`**

```tsx
const STAGES = [
  ["Ingest", "PDF / Markdown / URL -> token-aware chunker -> embeddings + BM25 + knowledge graph"],
  ["Retrieve", "hybrid vector + BM25 with RRF fusion, then GraphRAG expansion"],
  ["Rerank", "Cohere reranker narrows to the most relevant chunks"],
  ["Generate", "grounded, cited answer; refuses when context is insufficient; streams tokens"],
  ["Agent", "routes the query and self-corrects (rewrite + retry) when the answer is weak"],
];

export function AboutPage() {
  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-muted/30 bg-surface p-6">
        <h1 className="text-lg font-semibold text-primary">Production RAG</h1>
        <p className="mt-2 text-sm text-ink/80">
          Hybrid retrieval and reranking with GraphRAG, grounded and cited answers, and an agentic
          self-correction loop. Built as a provider-agnostic, observable, production-style RAG service.
        </p>
      </section>
      <section className="rounded-lg border border-muted/30 bg-surface p-6">
        <h2 className="mb-4 text-sm font-semibold text-ink">Pipeline</h2>
        <ol className="space-y-3">
          {STAGES.map(([name, desc], i) => (
            <li key={name} className="flex gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-xs text-white">
                {i + 1}
              </span>
              <div>
                <div className="font-medium text-ink">{name}</div>
                <div className="text-sm text-muted">{desc}</div>
              </div>
            </li>
          ))}
        </ol>
      </section>
      <p className="text-xs text-muted">
        API docs are served by the backend at <code>/docs</code>.
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Typecheck/build**

Run: `cd frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/AboutPage.tsx
git commit -m "feat: about/architecture page"
```

---

### Task 13: Final verification, README, deploy notes

**Files:**
- Modify: `README.md` (finalize the Demo UI section with build/deploy notes)

**Interfaces:**
- Produces: a fully green build/test run and accurate run/build/deploy docs.

- [ ] **Step 1: Full frontend verification**

Run: `cd frontend && npm test && npm run build`
Expected: all Vitest tests pass; `dist/` builds with no TypeScript errors.

- [ ] **Step 2: Full backend verification**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 3: Manual smoke test (documented, run by the user)**

Document these steps in the commit body or PR description; they require live services:
1. `docker-compose up -d` (Qdrant + API).
2. `cd frontend && npm run dev`; open `http://localhost:5173`.
3. Header status dot shows `ready`.
4. Documents page: upload a small `.md`, confirm it appears in the table; delete it.
5. Chat page: ask a question with streaming on; confirm tokens stream, sources and usage render.
6. Toggle Agent mode; confirm the reasoning trace and `route`/`attempts` appear.

- [ ] **Step 4: Finalize README Demo UI section**

Replace the Demo UI section body written in Task 3 with the full version:

```markdown
## Demo UI

A React single-page app (Vite + TypeScript) in `frontend/` exposes the full system: chat with live
token streaming and cited sources, agentic mode with a visible reasoning trace, document
upload/listing/deletion, and an architecture overview.

```bash
cd frontend
cp .env.example .env        # set VITE_API_URL to your backend (default http://localhost:8000)
npm install
npm run dev                 # http://localhost:5173

npm run build               # static assets in frontend/dist for Vercel/Netlify
```

The SPA calls the backend directly, so set the backend's `CORS_ORIGINS` to the SPA origin
(`http://localhost:5173` in dev). The public demo runs the backend with `API_KEY_HASH` unset.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: finalize React frontend run and deploy instructions"
```

---

## Self-Review

**Spec coverage:**
- No-orphan map -> every endpoint has a task: `/chat`,`/agent` (+stream) -> Tasks 8-10; `/ingest/upload` -> Task 1 + 11; `/ingest` URL -> Task 11; `GET`/`DELETE /ingest/documents` -> Task 11; `/health/*` -> Task 7; usage/guardrails/sources/route/attempts -> Tasks 8-9; architecture -> Task 12. Covered.
- Backend change limited to `/ingest/upload` + `.env.example` -> Tasks 1-2. Covered.
- Streamlit removal -> Task 3. Covered.
- Palette tokens + motion -> Task 4 (theme) + framer-motion across components. Covered.
- Static SPA deploy + VITE_API_URL -> Tasks 4, 13. Covered.
- Testing (stream parser, chat, documents, upload, apiClient, backend upload) -> Tasks 1, 5, 6, 7, 8, 9, 10, 11. Covered.

**Placeholder scan:** No TBD/TODO; every code step includes complete code; no "similar to Task N" references.

**Type consistency:** `ClientOptions = { baseUrl, apiKey? }` used identically in client/contexts/hooks. `StreamEvent` union fields (`node`, `sources`, `token`, `answer`/`usage`/`latency_ms`/`route`/`attempts`, `detail`) match the backend and are consumed consistently in `streamNdjson`/`useChat`. `ChatMessage` defined in Task 8, consumed in Task 9 `MessageThread`. `DocumentRecord`/`IngestResult` defined in Task 5, used in Task 11. Consistent.

One known cross-file note carried in the plan: the `useHealth` test uses JSX, so the file is `useHealth.test.tsx` (Step 1 of Task 7 states this).
