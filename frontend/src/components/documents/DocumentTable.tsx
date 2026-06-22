import { Trash2 } from "lucide-react";
import type { DocumentRecord } from "../../api/types";

export function DocumentTable({ docs, onDelete }: { docs: DocumentRecord[]; onDelete: (id: string) => void }) {
  if (docs.length === 0) return <p className="text-sm text-muted">No documents ingested yet.</p>;
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-xs uppercase text-muted">
        <tr>
          <th className="py-2">Source</th>
          <th className="py-2">Chunks</th>
          <th className="py-2">Ingested</th>
          <th className="py-2" />
        </tr>
      </thead>
      <tbody>
        {docs.map((d) => (
          <tr key={d.id} className="border-t border-muted/20">
            <td className="max-w-xs truncate py-2 font-mono text-xs">{d.source}</td>
            <td className="py-2">{d.chunks}</td>
            <td className="py-2 text-muted">{d.ingested_at}</td>
            <td className="py-2 text-right">
              <button aria-label={`Delete ${d.source}`} className="text-muted hover:text-danger" onClick={() => onDelete(d.id)}>
                <Trash2 size={16} />
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
