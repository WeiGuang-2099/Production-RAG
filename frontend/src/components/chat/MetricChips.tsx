import { ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";
import type { ChatMessage } from "../../hooks/useChat";

function Chip({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "amber" }) {
  const cls =
    tone === "amber"
      ? "border-highlight bg-highlight text-highlight-ink"
      : "border-line bg-surface text-muted";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[11px] ${cls}`}>
      {children}
    </span>
  );
}

export function MetricChips({ message, interpretedAs }: { message: ChatMessage; interpretedAs?: string }) {
  const { usage, latency_ms, route, attempts, guardrails } = message;
  const cost = usage?.cost_usd;
  const pii = guardrails?.pii_redacted ?? [];
  const flags = guardrails?.flags ?? [];
  const guardParts = [...pii.map((r) => `PII:${r}`), ...flags.map((f) => `flag:${f}`)];

  const chips: ReactNode[] = [];
  if (latency_ms != null) chips.push(<Chip key="lat">{Math.round(latency_ms)} ms</Chip>);
  if (typeof cost === "number") chips.push(<Chip key="cost">${cost.toFixed(5)}</Chip>);
  if (usage?.input_tokens != null)
    chips.push(<Chip key="tok">{usage.input_tokens}/{usage.output_tokens ?? 0} tok</Chip>);
  if (route)
    chips.push(<Chip key="route">route: {route} · {attempts ?? 0}</Chip>);
  if (guardParts.length > 0)
    chips.push(
      <Chip key="guard" tone="amber">
        <ShieldCheck size={11} /> {guardParts.join(", ")}
      </Chip>,
    );
  if (interpretedAs) chips.push(<Chip key="interp">interpreted as: {interpretedAs}</Chip>);

  if (chips.length === 0) return null;
  return <div className="mt-2 flex flex-wrap items-center gap-1.5">{chips}</div>;
}
