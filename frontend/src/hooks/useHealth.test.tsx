import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { useHealth } from "./useHealth";

afterEach(() => vi.restoreAllMocks());

const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>;

test("maps a ready response to ready", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "ready", checks: { app: "ok", qdrant: "ok" } }), { status: 200 }),
    ),
  );
  const { result } = renderHook(() => useHealth(), { wrapper });
  await waitFor(() => expect(result.current.status).toBe("ready"));
});

test("maps a failed fetch to down", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network")));
  const { result } = renderHook(() => useHealth(), { wrapper });
  await waitFor(() => expect(result.current.status).toBe("down"));
});
