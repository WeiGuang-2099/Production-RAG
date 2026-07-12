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
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <h1 className="font-serif text-2xl font-semibold text-ink">Documents</h1>
        <section className="rounded-lg border border-line bg-surface p-4 shadow-sm">
          <h2 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-primary">Add documents</h2>
          <UploadZone onFile={ingestFile} />
          <div className="mt-4">
            <UrlIngest onSubmit={ingestUrl} />
          </div>
        </section>
        <section className="rounded-lg border border-line bg-surface p-4 shadow-sm">
          <h2 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-primary">Ingested documents</h2>
          <DocumentTable docs={docs} onDelete={remove} />
        </section>
      </div>
    </div>
  );
}
