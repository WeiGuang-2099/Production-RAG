import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { ToastProvider } from "../context/ToastContext";
import { useChat } from "./useChat";

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

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

test("hydrates messages from localStorage and clear() empties them", async () => {
  localStorage.setItem(
    "test-chat",
    JSON.stringify([{ role: "user", content: "hi" }, { role: "assistant", content: "yo" }]),
  );
  const { result } = renderHook(() => useChat({ persistKey: "test-chat" }), { wrapper });
  expect(result.current.messages).toHaveLength(2);

  act(() => result.current.clear());
  expect(result.current.messages).toHaveLength(0);
  expect(localStorage.getItem("test-chat")).toBeNull();
});
