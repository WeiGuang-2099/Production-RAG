const STAGES = [
  ["Ingest", "PDF / Markdown / URL -> token-aware chunker -> embeddings + BM25 + knowledge graph"],
  ["Retrieve", "hybrid vector + BM25 with RRF fusion, then GraphRAG expansion"],
  ["Rerank", "Cohere reranker narrows to the most relevant chunks"],
  ["Generate", "grounded, cited answer; refuses when context is insufficient; streams tokens"],
  ["Agent", "routes the query and self-corrects (rewrite + retry) when the answer is weak"],
];

export function AboutPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <section>
          <h1 className="font-serif text-3xl font-semibold text-ink">Production RAG</h1>
          <p className="mt-3 leading-relaxed text-ink/80">
            Hybrid retrieval and reranking with GraphRAG, grounded and cited answers, and an agentic
            self-correction loop. Built as a provider-agnostic, observable, production-style RAG service.
          </p>
        </section>
        <section className="rounded-lg border border-line bg-surface p-6 shadow-sm">
          <h2 className="mb-4 text-[11px] font-medium uppercase tracking-wider text-primary">Pipeline</h2>
          <ol className="space-y-3">
            {STAGES.map(([name, desc], i) => (
              <li key={name} className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-highlight font-mono text-xs text-highlight-ink">
                  {i + 1}
                </span>
                <div>
                  <div className="font-medium text-ink">{name}</div>
                  <div className="text-sm text-muted">{desc}</div>
                </div>
              </li>
            ))}
          </ol>
        </section>
        <p className="text-xs text-muted">
          API docs are served by the backend at <code>/docs</code>.
        </p>
      </div>
    </div>
  );
}
