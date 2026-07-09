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
  onClear: () => void;
  onSend: (q: string) => void;
}

export function ChatControls({ agent, stream, topK, busy, onAgent, onStream, onTopK, onClear, onSend }: Props) {
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
        <button
          type="button"
          className="ml-auto rounded border border-muted/50 px-2 py-1 text-xs hover:bg-muted/10"
          onClick={onClear}
        >
          New chat
        </button>
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
