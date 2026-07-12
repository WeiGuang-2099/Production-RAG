import { motion } from "framer-motion";
import Markdown from "react-markdown";
import type { ChatMessage } from "../../hooks/useChat";
import { AgentTrace } from "./AgentTrace";
import { GuardrailsBadge } from "./GuardrailsBadge";
import { SourceCards } from "./SourceCards";
import { UsageBar } from "./UsageBar";

export function MessageThread({ messages }: { messages: ChatMessage[] }) {
  return (
    <div className="space-y-4">
      {messages.map((m, i) => (
        <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
          <div className={`text-xs font-medium ${m.role === "user" ? "text-primary" : "text-muted"}`}>{m.role}</div>
          {m.role === "assistant" && <AgentTrace steps={m.steps} route={m.route} attempts={m.attempts} />}
          {m.role === "assistant" && m.condensed_question && m.condensed_question !== messages[i - 1]?.content && (
            <div className="text-xs italic text-muted">interpreted as: {m.condensed_question}</div>
          )}
          <div className="prose prose-sm max-w-none text-ink">
            <Markdown>{m.content || "..."}</Markdown>
          </div>
          {m.role === "assistant" && (
            <>
              <GuardrailsBadge guardrails={m.guardrails} />
              <SourceCards sources={m.sources ?? []} />
              <UsageBar usage={m.usage} latency_ms={m.latency_ms} />
            </>
          )}
        </motion.div>
      ))}
    </div>
  );
}
