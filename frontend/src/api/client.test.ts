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
