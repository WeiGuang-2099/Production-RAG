import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
import { displaySource } from "../../api/sources";
import type { ChatMessage } from "../../hooks/useChat";

export interface CitationFocus {
  messageIndex: number;
  citation: number;
}

function lastSourcedIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant" && (messages[i].sources?.length ?? 0) > 0) return i;
  }
  return -1;
}

export function SourceInspector({
  messages,
  focused,
}: {
  messages: ChatMessage[];
  focused: CitationFocus | null;
}) {
  const index = focused?.messageIndex ?? lastSourcedIndex(messages);
  const sources = index >= 0 ? messages[index]?.sources ?? [] : [];
  const refs = useRef<Record<number, HTMLLIElement | null>>({});

  useEffect(() => {
    if (focused) refs.current[focused.citation]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [focused]);

  if (sources.length === 0) {
    return (
      <div className="p-4 text-sm text-muted">
        Sources appear here once an answer cites the corpus.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-line px-4 py-3 text-[11px] font-medium uppercase tracking-wider text-primary">
        Sources ({sources.length})
      </div>
      <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
        {sources.map((s, i) => {
          const citation = Number(s.metadata?.citation ?? i + 1);
          const isFocused = focused != null && focused.citation === citation;
          const score = s.metadata?.score;
          const raw = s.metadata?.source ?? "unknown";
          return (
            <motion.li
              key={`${index}-${citation}`}
              ref={(el) => { refs.current[citation] = el; }}
              data-testid={`source-card-${citation}`}
              data-focused={isFocused || undefined}
              animate={isFocused ? { backgroundColor: ["#FEF3C7", "#FFFFFF"] } : undefined}
              transition={{ duration: 1.2 }}
              className={`rounded-md border p-3 text-sm ${
                isFocused ? "border-primary" : "border-line"
              } bg-surface`}
            >
              <div className="flex items-center gap-2 font-mono text-xs text-muted">
                <span className="text-primary">[{citation}]</span>
                <span className="truncate" title={String(raw)}>
                  {displaySource(String(raw))}
                </span>
              </div>
              {typeof score === "number" && (
                <div className="mt-1.5 h-1 w-full rounded-full bg-sunken">
                  <div
                    className="h-1 rounded-full bg-primary"
                    style={{ width: `${Math.round(Math.min(1, Math.max(0, score)) * 100)}%` }}
                  />
                </div>
              )}
              <p className="mt-1.5 leading-relaxed text-ink/80">{s.content.slice(0, 400)}</p>
            </motion.li>
          );
        })}
      </ul>
    </div>
  );
}
