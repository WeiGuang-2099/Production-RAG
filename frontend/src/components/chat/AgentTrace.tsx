import { motion } from "framer-motion";

export function AgentTrace({ steps }: { steps?: string[] }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-1">
      {steps.map((s, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-full border border-line bg-sunken px-2 py-0.5 font-mono text-[11px] text-muted"
        >
          {s}
        </motion.span>
      ))}
    </div>
  );
}
