import { useEffect } from "react";
import { DocumentTable } from "../components/documents/DocumentTable";
import { UploadZone } from "../components/documents/UploadZone";
import { UrlIngest } from "../components/documents/UrlIngest";
import { useDocuments } from "../hooks/useDocuments";

export function DocumentsPage() {
  const { docs, refresh, remove, ingestUrl, ingestFile } = useDocuments();
  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-muted/30 bg-surface p-4">
        <h2 className="mb-3 text-sm font-semibold text-ink">Add documents</h2>
        <UploadZone onFile={ingestFile} />
        <div className="mt-4">
          <UrlIngest onSubmit={ingestUrl} />
        </div>
      </section>
      <section className="rounded-lg border border-muted/30 bg-surface p-4">
        <h2 className="mb-3 text-sm font-semibold text-ink">Ingested documents</h2>
        <DocumentTable docs={docs} onDelete={remove} />
      </section>
    </div>
  );
}
