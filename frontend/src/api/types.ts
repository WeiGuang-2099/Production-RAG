export interface SourceItem {
  content: string;
  metadata: { citation?: number | string; source?: string; score?: number } & Record<string, unknown>;
}
export interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  model?: string;
}
export interface Guardrails {
  pii_redacted?: string[];
  flags?: string[];
}
export interface ChatResult {
  answer: string;
  sources: SourceItem[];
  latency_ms: number;
  total_sources?: number;
  usage?: Usage;
  guardrails?: Guardrails;
  route?: string;
  attempts?: number;
}
export type StreamEvent =
  | { event: "step"; node: string }
  | { event: "sources"; sources: SourceItem[] }
  | { event: "token"; token: string }
  | {
      event: "done";
      answer: string;
      usage?: Usage;
      latency_ms?: number;
      guardrails?: Guardrails;
      route?: string;
      attempts?: number;
    }
  | { event: "error"; detail: string };
export interface DocumentRecord {
  id: string;
  source: string;
  chunks: number;
  ingested_at: string;
}
export interface IngestResult {
  source: string;
  chunks: number;
  status: string;
}
export interface HealthStatus {
  status: string;
  checks: Record<string, string>;
}
