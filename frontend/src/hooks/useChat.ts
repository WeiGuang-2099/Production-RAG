import { useCallback, useEffect, useState } from "react";
import { ApiError, postJson, postStream } from "../api/client";
import { streamNdjson } from "../api/stream";
import type { ChatResult, Guardrails, SourceItem, Usage } from "../api/types";
import { useSettings } from "../context/SettingsContext";
import { useToast } from "../context/ToastContext";
import {
  addSession, loadStore, removeSession, saveStore, upsertActiveMessages,
  type SessionStore,
} from "../state/sessions";

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
  condensed_question?: string;
}

export interface SendOpts {
  agent: boolean;
  stream: boolean;
  topK: number;
  sources?: string[];
}

function activeMessages(store: SessionStore): ChatMessage[] {
  return store.sessions.find((s) => s.id === store.activeId)?.messages ?? [];
}

export function useChat() {
  const { client } = useSettings();
  const { toast } = useToast();
  const [store, setStore] = useState<SessionStore>(() => loadStore());
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    saveStore(store);
  }, [store]);

  const messages = activeMessages(store);

  const newSession = useCallback(() => {
    // No-op when the active session is still empty - never spawn blanks.
    setStore((st) => (activeMessages(st).length === 0 ? st : addSession(st)));
  }, []);

  const switchSession = useCallback((id: string) => {
    setStore((st) => (st.sessions.some((s) => s.id === id) ? { ...st, activeId: id } : st));
  }, []);

  const deleteSession = useCallback((id: string) => {
    setStore((st) => removeSession(st, id));
  }, []);

  const patchLast = useCallback((patch: Partial<ChatMessage> | ((m: ChatMessage) => ChatMessage)) => {
    setStore((st) => {
      const ms = activeMessages(st).slice();
      const i = ms.length - 1;
      if (i < 0) return st;
      ms[i] = typeof patch === "function" ? patch(ms[i]) : { ...ms[i], ...patch };
      return upsertActiveMessages(st, ms);
    });
  }, []);

  const send = useCallback(
    async (question: string, opts: SendOpts) => {
      if (!question.trim() || busy) return;
      setBusy(true);
      setStore((st) =>
        upsertActiveMessages(st, [
          ...activeMessages(st),
          { role: "user", content: question },
          { role: "assistant", content: "" },
        ]),
      );
      const history = messages
        .filter((m) => m.content && !m.content.startsWith("Error:"))
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));
      const base = opts.agent ? "/agent" : "/chat";
      const body: {
        question: string;
        top_k: number;
        sources?: string[];
        history?: { role: "user" | "assistant"; content: string }[];
      } = { question, top_k: opts.topK };
      if (opts.sources && opts.sources.length > 0) body.sources = opts.sources;
      if (history.length > 0) body.history = history;
      try {
        if (opts.stream) {
          const stream = await postStream(client, `${base}/stream`, body);
          for await (const ev of streamNdjson(stream)) {
            if (ev.event === "step") patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), ev.node] }));
            else if (ev.event === "condensed") patchLast({ condensed_question: ev.condensed_question });
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
            condensed_question: r.condensed_question ?? undefined,
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
    [busy, client, messages, patchLast, toast],
  );

  return {
    sessions: store.sessions,
    activeId: store.activeId,
    messages,
    busy,
    send,
    newSession,
    switchSession,
    deleteSession,
    // Transitional alias, removed in the chat-column task:
    clear: newSession,
  };
}
