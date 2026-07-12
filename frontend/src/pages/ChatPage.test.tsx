import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, vi } from "vitest";
import { ChatProvider } from "../context/ChatContext";
import { SettingsProvider } from "../context/SettingsContext";
import { ToastProvider } from "../context/ToastContext";
import { ChatPage } from "./ChatPage";

// Isolate the chat stream's fetch stub from the documents fetch on mount
vi.mock("../hooks/useDocuments", () => ({
  useDocuments: () => ({ docs: [], refresh: vi.fn() }),
}));

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

function ui() {
  return render(
    <SettingsProvider>
      <ToastProvider>
        <ChatProvider>
          <ChatPage />
        </ChatProvider>
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

test("empty state example question sends immediately", async () => {
  const enc = new TextEncoder();
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(
      new ReadableStream({
        start(c) {
          c.enqueue(enc.encode('{"event":"done","answer":"ok"}\n'));
          c.close();
        },
      }),
      { status: 200 },
    ),
  );
  vi.stubGlobal("fetch", fetchMock);
  ui();
  await userEvent.click(
    screen.getByRole("button", { name: /What does the Transformer architecture eliminate/i }),
  );
  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  const body = JSON.parse(fetchMock.mock.calls[0][1].body);
  expect(body.question).toMatch(/Transformer architecture eliminate/);
});

test("mode segmented control switches to the agent endpoint", async () => {
  const enc = new TextEncoder();
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(
      new ReadableStream({
        start(c) {
          c.enqueue(enc.encode('{"event":"done","answer":"ok"}\n'));
          c.close();
        },
      }),
      { status: 200 },
    ),
  );
  vi.stubGlobal("fetch", fetchMock);
  ui();
  await userEvent.click(screen.getByRole("button", { name: /^agent$/i }));
  await userEvent.type(screen.getByPlaceholderText(/Ask a question/i), "hello");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  expect(fetchMock.mock.calls[0][0]).toMatch(/\/agent\/stream$/);
});
