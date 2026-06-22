import { ShieldCheck } from "lucide-react";
import type { Guardrails } from "../../api/types";

export function GuardrailsBadge({ guardrails }: { guardrails?: Guardrails }) {
  const redacted = guardrails?.pii_redacted ?? [];
  const flags = guardrails?.flags ?? [];
  if (redacted.length === 0 && flags.length === 0) return null;
  const parts = [...redacted.map((r) => `PII:${r}`), ...flags.map((f) => `flag:${f}`)];
  return (
    <span className="mt-2 inline-flex items-center gap-1 rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-800">
      <ShieldCheck size={12} />
      {parts.join(", ")}
    </span>
  );
}
