const EXAMPLES = [
  "What does the Transformer architecture eliminate, and what does it rely on instead?",
  "What are the two pre-training objectives used by BERT?",
  "By what factor can LoRA reduce trainable parameters on GPT-3 175B?",
];

export function EmptyState({ onAsk }: { onAsk: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 text-center">
      <h1 className="font-serif text-3xl font-semibold text-ink">Ask the papers.</h1>
      <p className="mt-2 max-w-md text-sm text-muted">
        Hybrid retrieval over the ingested corpus with grounded, cited answers — the model
        refuses when the documents do not contain the answer.
      </p>
      <div className="mt-6 flex w-full max-w-md flex-col gap-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            type="button"
            className="rounded-lg border border-line bg-surface px-4 py-3 text-left text-sm text-ink shadow-sm hover:border-primary hover:text-primary"
            onClick={() => onAsk(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
