import { useState } from "react";
import { ChatControls } from "../components/chat/ChatControls";
import { MessageThread } from "../components/chat/MessageThread";
import { useChat } from "../hooks/useChat";

export function ChatPage() {
  const { messages, busy, send } = useChat();
  const [agent, setAgent] = useState(false);
  const [stream, setStream] = useState(true);
  const [topK, setTopK] = useState(5);

  return (
    <div className="flex h-[calc(100vh-9rem)] flex-col rounded-lg border border-muted/30 bg-surface">
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
        onSend={(q) => send(q, { agent, stream, topK })}
      />
    </div>
  );
}
