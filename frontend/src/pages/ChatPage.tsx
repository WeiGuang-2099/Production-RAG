import { useEffect } from "react";
import { Composer } from "../components/chat/Composer";
import { EmptyState } from "../components/chat/EmptyState";
import { MessageThread } from "../components/chat/MessageThread";
import { SourceInspector } from "../components/chat/SourceInspector";
import { useChatContext } from "../context/ChatContext";
import { useDocuments } from "../hooks/useDocuments";

export function ChatPage() {
  const { messages, busy, send, agent, stream, topK, setAgent, setStream, setTopK,
    scopeSources, setScopeSources, focused, setFocused } = useChatContext();
  const { docs, refresh } = useDocuments();
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { setFocused(null); }, [messages.length, setFocused]);

  const ask = (q: string) => send(q, { agent, stream, topK, sources: scopeSources });

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState onAsk={ask} />
          ) : (
            <div className="mx-auto max-w-3xl px-4 py-8">
              <MessageThread
                messages={messages}
                onCitation={(messageIndex, citation) => setFocused({ messageIndex, citation })}
              />
            </div>
          )}
        </div>
        <Composer
          agent={agent}
          stream={stream}
          topK={topK}
          busy={busy}
          docs={docs}
          scopeSources={scopeSources}
          onAgent={setAgent}
          onStream={setStream}
          onTopK={setTopK}
          onScope={setScopeSources}
          onSend={ask}
        />
      </div>
      <aside className="hidden w-[320px] shrink-0 border-l border-line bg-surface lg:block">
        <SourceInspector messages={messages} focused={focused} />
      </aside>
    </div>
  );
}
