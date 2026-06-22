import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { SettingsProvider } from "../context/SettingsContext";
import { ToastProvider } from "../context/ToastContext";
import { useDocuments } from "./useDocuments";

afterEach(() => vi.restoreAllMocks());

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <SettingsProvider>
    <ToastProvider>{children}</ToastProvider>
  </SettingsProvider>
);

test("refresh loads documents", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify([{ id: "a", source: "doc.pdf", chunks: 3, ingested_at: "2026-06-22" }]), {
        status: 200,
      }),
    ),
  );
  const { result } = renderHook(() => useDocuments(), { wrapper });
  await act(async () => result.current.refresh());
  await waitFor(() => expect(result.current.docs).toHaveLength(1));
  expect(result.current.docs[0].source).toBe("doc.pdf");
});
