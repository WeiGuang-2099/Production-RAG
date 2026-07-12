import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { ChatProvider } from "../../context/ChatContext";
import { SettingsProvider } from "../../context/SettingsContext";
import { ToastProvider } from "../../context/ToastContext";
import { SessionRail } from "./SessionRail";

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

function ui() {
  return render(
    <MemoryRouter>
      <SettingsProvider>
        <ToastProvider>
          <ChatProvider>
            <SessionRail />
          </ChatProvider>
        </ToastProvider>
      </SettingsProvider>
    </MemoryRouter>,
  );
}

test("lists sessions from the store and switches on click", async () => {
  localStorage.setItem(
    "rag-chats",
    JSON.stringify({
      activeId: "b",
      sessions: [
        { id: "a", title: "What is LoRA?", createdAt: 1, updatedAt: 1,
          messages: [{ role: "user", content: "What is LoRA?" }] },
        { id: "b", title: "What is BERT?", createdAt: 2, updatedAt: 2,
          messages: [{ role: "user", content: "What is BERT?" }] },
      ],
    }),
  );
  ui();
  expect(screen.getByText("What is LoRA?")).toBeInTheDocument();
  expect(screen.getByText("What is BERT?")).toBeInTheDocument();
  await userEvent.click(screen.getByText("What is LoRA?"));
  expect(JSON.parse(localStorage.getItem("rag-chats")!).activeId).toBe("a");
});

test("has brand, New chat, and the bottom navigation", () => {
  // Seed a real-titled session so the New chat *action* button is the only
  // element named "New chat" (a fresh store's default session is titled that).
  localStorage.setItem(
    "rag-chats",
    JSON.stringify({
      activeId: "a",
      sessions: [
        { id: "a", title: "What is LoRA?", createdAt: 1, updatedAt: 1,
          messages: [{ role: "user", content: "What is LoRA?" }] },
      ],
    }),
  );
  ui();
  expect(screen.getByText("Production RAG")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /new chat/i })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /documents/i })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /about/i })).toBeInTheDocument();
});

test("delete button removes a session", async () => {
  localStorage.setItem(
    "rag-chats",
    JSON.stringify({
      activeId: "a",
      sessions: [
        { id: "a", title: "Thread A", createdAt: 1, updatedAt: 1,
          messages: [{ role: "user", content: "Thread A" }] },
        { id: "b", title: "Thread B", createdAt: 2, updatedAt: 2,
          messages: [{ role: "user", content: "Thread B" }] },
      ],
    }),
  );
  ui();
  await userEvent.click(screen.getByRole("button", { name: /delete thread b/i }));
  expect(screen.queryByText("Thread B")).toBeNull();
});
