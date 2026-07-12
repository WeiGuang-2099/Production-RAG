import { displaySource } from "../../api/sources";
import type { DocumentRecord } from "../../api/types";

interface Props {
  docs: DocumentRecord[];
  selected: string[];
  onChange: (next: string[]) => void;
}

export function DocumentScopePicker({ docs, selected, onChange }: Props) {
  if (docs.length === 0) return null;
  const toggle = (source: string) => {
    onChange(
      selected.includes(source) ? selected.filter((s) => s !== source) : [...selected, source],
    );
  };
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-muted">Scope:</span>
      <button
        type="button"
        className={`rounded-full border px-2 py-0.5 ${selected.length === 0 ? "border-primary bg-highlight text-highlight-ink" : "border-line text-muted hover:text-ink"}`}
        onClick={() => onChange([])}
      >
        All documents
      </button>
      {docs.map((d) => {
        const on = selected.includes(d.source);
        return (
          <button
            key={d.id}
            type="button"
            title={d.source}
            className={`rounded-full border px-2 py-0.5 ${on ? "border-primary bg-highlight text-highlight-ink" : "border-line text-muted hover:text-ink"}`}
            onClick={() => toggle(d.source)}
          >
            {displaySource(d.source)}
          </button>
        );
      })}
    </div>
  );
}
