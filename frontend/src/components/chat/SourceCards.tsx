import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { displaySource } from "../../api/sources";
import type { SourceItem } from "../../api/types";

export function SourceCards({ sources }: { sources: SourceItem[] }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-3">
      <button className="text-sm text-primary hover:underline" onClick={() => setOpen((o) => !o)}>
        Sources ({sources.length})
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 space-y-2 overflow-hidden"
          >
            {sources.map((s, i) => {
              const citation = s.metadata?.citation ?? i + 1;
              const score = s.metadata?.score;
              return (
                <li key={i} className="rounded-md border border-muted/30 bg-surface p-3 text-sm">
                  <div className="flex items-center gap-2 font-mono text-xs text-muted">
                    <span className="text-primary">[{String(citation)}]</span>
                    <span className="truncate" title={s.metadata?.source ?? "unknown"}>
                      {displaySource(s.metadata?.source ?? "unknown")}
                    </span>
                    {typeof score === "number" && <span>· {score.toFixed(2)}</span>}
                  </div>
                  <p className="mt-1 text-ink/80">{s.content.slice(0, 500)}</p>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
