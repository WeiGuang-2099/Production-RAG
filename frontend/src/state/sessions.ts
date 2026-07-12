// Pure multi-session persistence for the chat workbench. Schema v2 under
// STORE_KEY; a legacy single-conversation array under LEGACY_KEY is
// migrated once and removed. Any parse failure falls back to a fresh
// store - persistence must never brick the page.
import type { ChatMessage } from "../hooks/useChat";

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface SessionStore {
  activeId: string;
  sessions: ChatSession[];
}

export const STORE_KEY = "rag-chats";
export const LEGACY_KEY = "rag-chat";
export const MAX_SESSIONS = 20;
export const MAX_MESSAGES = 50;

function newSessionObject(): ChatSession {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function titleFrom(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === "user" && m.content);
  return first ? first.content.slice(0, 60) : "New chat";
}

export function loadStore(): SessionStore {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as SessionStore;
      if (parsed && Array.isArray(parsed.sessions) && parsed.sessions.length > 0) return parsed;
    }
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const messages = JSON.parse(legacy) as ChatMessage[];
      localStorage.removeItem(LEGACY_KEY);
      if (Array.isArray(messages) && messages.length > 0) {
        const s = { ...newSessionObject(), title: titleFrom(messages), messages };
        return { activeId: s.id, sessions: [s] };
      }
    }
  } catch {
    /* corrupt storage: fall through to a fresh store */
  }
  const s = newSessionObject();
  return { activeId: s.id, sessions: [s] };
}

export function saveStore(store: SessionStore): void {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(store));
  } catch {
    /* quota exceeded: keep in-memory state */
  }
}

export function upsertActiveMessages(store: SessionStore, messages: ChatMessage[]): SessionStore {
  return {
    ...store,
    sessions: store.sessions.map((s) =>
      s.id === store.activeId
        ? {
            ...s,
            messages: messages.slice(-MAX_MESSAGES),
            title: titleFrom(messages),
            updatedAt: Date.now(),
          }
        : s,
    ),
  };
}

export function addSession(store: SessionStore): SessionStore {
  const s = newSessionObject();
  let sessions = [s, ...store.sessions];
  if (sessions.length > MAX_SESSIONS) {
    // Evict the stalest of the pre-existing sessions only - never the one we
    // just added (a tight loop can give every session the same Date.now(),
    // and a stable sort would otherwise pick the freshly-prepended session).
    const stalest = [...store.sessions].sort((a, b) => a.updatedAt - b.updatedAt)[0];
    sessions = sessions.filter((x) => x.id !== stalest.id);
  }
  return { activeId: s.id, sessions };
}

export function removeSession(store: SessionStore, id: string): SessionStore {
  let sessions = store.sessions.filter((s) => s.id !== id);
  if (sessions.length === 0) {
    const s = newSessionObject();
    return { activeId: s.id, sessions: [s] };
  }
  const activeId = store.activeId === id ? sessions[0].id : store.activeId;
  return { activeId, sessions };
}
