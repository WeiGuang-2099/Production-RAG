import { motion } from "framer-motion";

export function AgentTrace({ steps, route, attempts }: { steps?: string[]; route?: string; attempts?: number }) {
  if ((!steps || steps.length === 0) && !route) return null;
  return (
    <div className="mb-2 rounded-md border border-muted/30 bg-bg px-3 py-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {route && <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-primary">route: {route}</span>}
        {typeof attempts === "number" && <span className="text-muted">attempts: {attempts}</span>}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {(steps ?? []).map((s, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            className="rounded bg-surface px-2 py-0.5 font-mono text-[11px] text-ink/70"
          >
            {s}
          </motion.span>
        ))}
      </div>
    </div>
  );
}
