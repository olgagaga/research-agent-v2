import type { SoloData, SessionInfo, ParallelData, ParallelRunInfo } from "./types";

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

export const api = {
  config: () => get<Record<string, string>>("/api/config"),
  sessions: () => get<SessionInfo[]>("/api/sessions"),
  solo: (session?: string, all = false) =>
    get<SoloData>(
      `/api/solo?${new URLSearchParams({
        ...(session ? { session } : {}),
        ...(all ? { all: "true" } : {}),
      })}`,
    ),
  parallelRuns: () => get<ParallelRunInfo[]>("/api/parallel"),
  parallel: (name: string) => get<ParallelData>(`/api/parallel/${encodeURIComponent(name)}`),
};
