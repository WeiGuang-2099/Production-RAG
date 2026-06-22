import type { Usage } from "../../api/types";

export function UsageBar({ usage, latency_ms }: { usage?: Usage; latency_ms?: number }) {
  if (!usage && latency_ms == null) return null;
  const cell = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-muted">{label}</span>
      <span className="font-mono text-sm text-ink">{value}</span>
    </div>
  );
  const cost = usage?.cost_usd;
  return (
    <div className="mt-2 flex gap-6 rounded-md bg-bg px-3 py-2">
      {cell("in", String(usage?.input_tokens ?? "-"))}
      {cell("out", String(usage?.output_tokens ?? "-"))}
      {cell("cost", typeof cost === "number" ? `$${cost.toFixed(5)}` : "-")}
      {cell("latency", latency_ms != null ? `${Math.round(latency_ms)} ms` : "-")}
    </div>
  );
}
