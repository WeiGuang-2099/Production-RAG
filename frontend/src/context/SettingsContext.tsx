import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ClientOptions } from "../api/client";

interface SettingsValue {
  client: ClientOptions;
  setBaseUrl: (s: string) => void;
  setApiKey: (s: string) => void;
}

const DEFAULT_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const SettingsContext = createContext<SettingsValue | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [baseUrl, setBaseUrlState] = useState(() => localStorage.getItem("rag.baseUrl") ?? DEFAULT_BASE);
  const [apiKey, setApiKeyState] = useState(() => localStorage.getItem("rag.apiKey") ?? "");

  const setBaseUrl = useCallback((s: string) => {
    setBaseUrlState(s);
    localStorage.setItem("rag.baseUrl", s);
  }, []);
  const setApiKey = useCallback((s: string) => {
    setApiKeyState(s);
    localStorage.setItem("rag.apiKey", s);
  }, []);

  const value = useMemo<SettingsValue>(
    () => ({ client: { baseUrl, apiKey: apiKey || undefined }, setBaseUrl, setApiKey }),
    [baseUrl, apiKey, setBaseUrl, setApiKey],
  );
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): SettingsValue {
  const v = useContext(SettingsContext);
  if (!v) throw new Error("useSettings must be used within SettingsProvider");
  return v;
}
