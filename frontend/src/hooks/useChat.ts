import { useCallback, useEffect, useState } from "react";
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

const MAX_PERSISTED = 50;

export function useChat(options: { persistKey?: string } = {}) {
  const { persistKey } = options;
  const { client } = useSettings();
  const { toast } = useToast();
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (!persistKey) return [];
    try {
      const raw = localStorage.getItem(persistKey);
      return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
    } catch {
      return [];
    }
  });
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!persistKey) return;
    try {
      // An empty thread removes the key so "New chat" leaves no residue
      if (messages.length === 0) {
        localStorage.removeItem(persistKey);
      } else {
        localStorage.setItem(persistKey, JSON.stringify(messages.slice(-MAX_PERSISTED)));
      }
    } catch {
      /* ignore quota/serialization errors; keep in-memory state */
    }
  }, [messages, persistKey]);

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

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

  return { messages, busy, send, clear };
}
