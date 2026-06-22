import { useHealth } from "../hooks/useHealth";

const COLOR: Record<string, string> = {
  loading: "bg-muted",
  ready: "bg-emerald-500",
  degraded: "bg-amber-500",
  down: "bg-danger",
};

export function StatusDot() {
  const { status, checks } = useHealth();
  const tip = Object.entries(checks)
    .map(([k, v]) => `${k}: ${v}`)
    .join("\n");
  return (
    <span className="flex items-center gap-2 text-xs text-muted" title={tip || status}>
      <span className={`h-2.5 w-2.5 rounded-full ${COLOR[status]}`} />
      {status}
    </span>
  );
}
