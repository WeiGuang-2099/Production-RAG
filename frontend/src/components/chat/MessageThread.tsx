import { motion } from "framer-motion";
import Markdown from "react-markdown";
import { linkCitations } from "../../api/citations";
import type { ChatMessage } from "../../hooks/useChat";
import { AgentTrace } from "./AgentTrace";
import { MetricChips } from "./MetricChips";
import { SourceCards } from "./SourceCards";

export function MessageThread({
  messages,
  onCitation,
}: {
  messages: ChatMessage[];
  onCitation?: (messageIndex: number, citation: number) => void;
}) {
  return (
    <div className="space-y-8">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <motion.h2
            key={i}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="font-serif text-xl font-semibold text-ink"
          >
            {m.content}
          </motion.h2>
        ) : (
          <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-primary">
              Answer{m.sources ? ` · ${m.sources.length} sources` : ""}
            </div>
            <AgentTrace steps={m.steps} />
            <div className="prose prose-sm max-w-none text-ink">
              <Markdown
                urlTransform={(url) => url}
                components={{
                  a: ({ href, children }) =>
                    href?.startsWith("citation:") ? (
                      <button
                        type="button"
                        className="mx-0.5 rounded bg-highlight px-1 font-mono text-[11px] text-highlight-ink hover:outline hover:outline-1 hover:outline-primary"
                        onClick={() => onCitation?.(i, Number(href.slice("citation:".length)))}
                      >
                        {children}
                      </button>
                    ) : (
                      <a href={href} target="_blank" rel="noreferrer">
                        {children}
                      </a>
                    ),
                }}
              >
                {linkCitations(m.content) || "..."}
              </Markdown>
            </div>
            <MetricChips
              message={m}
              interpretedAs={
                m.condensed_question && m.condensed_question !== messages[i - 1]?.content
                  ? m.condensed_question
                  : undefined
              }
            />
            {/* Inline sources are the <lg fallback; the inspector owns lg+ */}
            <div className="lg:hidden">
              <SourceCards sources={m.sources ?? []} />
            </div>
          </motion.div>
        ),
      )}
    </div>
  );
}
