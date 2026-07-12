import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { useChat } from "../hooks/useChat";

type ChatContextValue = ReturnType<typeof useChat> & {
  agent: boolean;
  stream: boolean;
  topK: number;
  scopeSources: string[];
  setAgent: (v: boolean) => void;
  setStream: (v: boolean) => void;
  setTopK: (v: number) => void;
  setScopeSources: (v: string[]) => void;
};

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const chat = useChat();
  const [agent, setAgent] = useState(false);
  const [stream, setStream] = useState(true);
  const [topK, setTopK] = useState(5);
  const [scopeSources, setScopeSources] = useState<string[]>([]);
  const value = useMemo(
    () => ({ ...chat, agent, stream, topK, scopeSources, setAgent, setStream, setTopK, setScopeSources }),
    [chat, agent, stream, topK, scopeSources],
  );
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChatContext(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used within ChatProvider");
  return ctx;
}
