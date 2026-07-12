import { Minus, Plus, Send, Zap } from "lucide-react";
import { useState } from "react";
import type { DocumentRecord } from "../../api/types";
import { DocumentScopePicker } from "./DocumentScopePicker";

interface Props {
  agent: boolean;
  stream: boolean;
  topK: number;
  busy: boolean;
  docs: DocumentRecord[];
  scopeSources: string[];
  onAgent: (v: boolean) => void;
  onStream: (v: boolean) => void;
  onTopK: (v: number) => void;
  onScope: (v: string[]) => void;
  onSend: (q: string) => void;
}

export function Composer({
  agent, stream, topK, busy, docs, scopeSources,
  onAgent, onStream, onTopK, onScope, onSend,
}: Props) {
  const [text, setText] = useState("");
  const submit = () => {
    if (!text.trim() || busy) return;
    onSend(text);
    setText("");
  };
  const seg = (on: boolean) =>
    `rounded-full px-3 py-1 text-xs ${on ? "bg-highlight text-highlight-ink" : "text-muted hover:text-ink"}`;

  return (
    <div className="border-t border-line bg-bg px-4 py-3">
      <div className="mx-auto max-w-3xl">
        <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
          <div role="group" aria-label="Mode" className="flex rounded-full border border-line bg-surface p-0.5">
            <button type="button" className={seg(!agent)} onClick={() => onAgent(false)}>
              Standard
            </button>
            <button type="button" className={seg(agent)} onClick={() => onAgent(true)}>
              Agent
            </button>
          </div>
          <button
            type="button"
            aria-label="Toggle streaming"
            aria-pressed={stream}
            title={stream ? "Streaming on" : "Streaming off"}
            className={`flex items-center gap-1 rounded-full border px-2.5 py-1 ${
              stream ? "border-highlight bg-highlight text-highlight-ink" : "border-line text-muted hover:text-ink"
            }`}
            onClick={() => onStream(!stream)}
          >
            <Zap size={12} /> stream
          </button>
          <div className="flex items-center gap-1 text-muted">
            <span>top_k</span>
            <button
              type="button" aria-label="Decrease top_k"
              className="rounded-full border border-line p-0.5 hover:text-ink"
              onClick={() => onTopK(Math.max(1, topK - 1))}
            >
              <Minus size={11} />
            </button>
            <span className="w-5 text-center font-mono text-ink">{topK}</span>
            <button
              type="button" aria-label="Increase top_k"
              className="rounded-full border border-line p-0.5 hover:text-ink"
              onClick={() => onTopK(Math.min(50, topK + 1))}
            >
              <Plus size={11} />
            </button>
          </div>
          <DocumentScopePicker docs={docs} selected={scopeSources} onChange={onScope} />
        </div>
        <div className="flex items-end gap-2">
          <textarea
            className="max-h-40 min-h-[2.75rem] flex-1 resize-none rounded-lg border border-line bg-surface px-3 py-2.5 text-sm shadow-sm focus:border-primary focus:outline-none"
            placeholder="Ask a question about the ingested documents"
            rows={1}
            value={text}
            disabled={busy}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <button
            className="flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2.5 text-sm text-white hover:bg-primary-hover disabled:opacity-50"
            disabled={busy}
            onClick={submit}
          >
            <Send size={14} /> Send
          </button>
        </div>
      </div>
    </div>
  );
}
