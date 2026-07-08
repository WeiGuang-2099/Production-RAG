// Maps a raw retrieval source value (path, URL, or the "graph" sentinel)
// to a short, human-readable label for the UI. Pure and dependency-free.
export function displaySource(raw: string): string {
  if (!raw) return "unknown";
  if (raw === "graph") return "Knowledge graph";
  if (/^https?:\/\//i.test(raw)) {
    try {
      const u = new URL(raw);
      const last = u.pathname.split("/").filter(Boolean).pop();
      return last ? `${u.host}/${last}` : u.host;
    } catch {
      return raw;
    }
  }
  const parts = raw.split(/[/\\]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : raw;
}
