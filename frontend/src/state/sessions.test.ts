import { afterEach, expect, test } from "vitest";
import {
  addSession, LEGACY_KEY, loadStore, MAX_SESSIONS, removeSession,
  STORE_KEY, titleFrom, upsertActiveMessages,
} from "./sessions";

afterEach(() => localStorage.clear());

test("loadStore migrates a legacy v1 conversation into the first session", () => {
  localStorage.setItem(
    LEGACY_KEY,
    JSON.stringify([
      { role: "user", content: "What is LoRA?" },
      { role: "assistant", content: "LoRA is ..." },
    ]),
  );
  const store = loadStore();
  expect(store.sessions).toHaveLength(1);
  expect(store.sessions[0].messages).toHaveLength(2);
  expect(store.sessions[0].title).toBe("What is LoRA?");
  expect(store.activeId).toBe(store.sessions[0].id);
  expect(localStorage.getItem(LEGACY_KEY)).toBeNull();
});

test("loadStore falls back to a fresh store on corrupt JSON", () => {
  localStorage.setItem(STORE_KEY, "{not json");
  const store = loadStore();
  expect(store.sessions).toHaveLength(1);
  expect(store.sessions[0].messages).toEqual([]);
});

test("titleFrom uses the first user message, truncated to 60 chars", () => {
  expect(titleFrom([{ role: "user", content: "x".repeat(100) }])).toBe("x".repeat(60));
  expect(titleFrom([])).toBe("New chat");
});

test("upsertActiveMessages caps messages and retitles the active session", () => {
  let store = loadStore();
  const many = Array.from({ length: 60 }, (_, i) => ({
    role: "user" as const, content: `m${i}`,
  }));
  store = upsertActiveMessages(store, many);
  expect(store.sessions[0].messages).toHaveLength(50);
  expect(store.sessions[0].title).toBe("m0");
});

test("addSession prepends and evicts the stalest beyond MAX_SESSIONS", () => {
  let store = loadStore();
  for (let i = 0; i < MAX_SESSIONS + 3; i++) store = addSession(store);
  expect(store.sessions.length).toBe(MAX_SESSIONS);
  expect(store.activeId).toBe(store.sessions[0].id);
});

test("removeSession of the active session activates the next, never empties", () => {
  let store = loadStore();
  const first = store.activeId;
  store = addSession(store);
  store = removeSession(store, store.activeId);
  expect(store.activeId).toBe(first);
  store = removeSession(store, first);
  expect(store.sessions).toHaveLength(1); // fresh replacement
  expect(store.sessions[0].messages).toEqual([]);
});
