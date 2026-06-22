import { act, renderHook } from "@testing-library/react";
import { SettingsProvider, useSettings } from "./SettingsContext";

test("defaults baseUrl and persists overrides", () => {
  localStorage.clear();
  const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>;
  const { result } = renderHook(() => useSettings(), { wrapper });
  expect(result.current.client.baseUrl).toBe("http://localhost:8000");
  act(() => result.current.setApiKey("k"));
  expect(result.current.client.apiKey).toBe("k");
  expect(localStorage.getItem("rag.apiKey")).toBe("k");
});
