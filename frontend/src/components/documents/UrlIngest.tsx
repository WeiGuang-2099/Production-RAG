import { useState } from "react";

export function UrlIngest({ onSubmit }: { onSubmit: (url: string) => void }) {
  const [url, setUrl] = useState("");
  return (
    <div className="flex gap-2">
      <input
        className="flex-1 rounded-md border border-muted/50 px-3 py-2 text-sm"
        placeholder="https://example.com/document.pdf"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <button
        className="rounded-md bg-primary px-4 py-2 text-sm text-white hover:bg-primary-hover disabled:opacity-50"
        disabled={!url.trim()}
        onClick={() => {
          onSubmit(url.trim());
          setUrl("");
        }}
      >
        Ingest URL
      </button>
    </div>
  );
}
