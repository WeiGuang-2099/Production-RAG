import { useEffect, useState } from "react";
import { getJson } from "../api/client";
import type { HealthStatus } from "../api/types";
import { useSettings } from "../context/SettingsContext";

type Status = "loading" | "ready" | "degraded" | "down";

export function useHealth(): { status: Status; checks: Record<string, string> } {
  const { client } = useSettings();
  const [status, setStatus] = useState<Status>("loading");
  const [checks, setChecks] = useState<Record<string, string>>({});

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const data = await getJson<HealthStatus>(client, "/health/ready");
        if (!active) return;
        setChecks(data.checks ?? {});
        const anyFailed = Object.values(data.checks ?? {}).some((v) => v.includes("failed") || v === "empty");
        setStatus(data.status === "ready" ? (anyFailed ? "degraded" : "ready") : "degraded");
      } catch {
        if (active) setStatus("down");
      }
    }
    poll();
    const id = setInterval(poll, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [client]);

  return { status, checks };
}
