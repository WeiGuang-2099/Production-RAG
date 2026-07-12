import { useEffect } from "react";
import { ChatControls } from "../components/chat/ChatControls";
import { DocumentScopePicker } from "../components/chat/DocumentScopePicker";
import { MessageThread } from "../components/chat/MessageThread";
import { useChatContext } from "../context/ChatContext";
import { useDocuments } from "../hooks/useDocuments";

export function ChatPage() {
  const { messages, busy, send, newSession, agent, stream, topK, setAgent, setStream, setTopK,
    scopeSources, setScopeSources } = useChatContext();
  const { docs, refresh } = useDocuments();
  useEffect(() => { refresh(); }, [refresh]);

  return (
    <div className="flex h-[calc(100vh-9rem)] flex-col rounded-lg border border-muted/30 bg-surface">
      <div className="border-b border-muted/30 px-4 pt-3">
        <DocumentScopePicker docs={docs} selected={scopeSources} onChange={setScopeSources} />
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted">Ask a question about your ingested documents.</p>
        ) : (
          <MessageThread messages={messages} />
        )}
      </div>
      <ChatControls
        agent={agent}
        stream={stream}
        topK={topK}
        busy={busy}
        onAgent={setAgent}
        onStream={setStream}
        onTopK={setTopK}
        onClear={newSession}
        onSend={(q) => send(q, { agent, stream, topK, sources: scopeSources })}
      />
    </div>
  );
}
