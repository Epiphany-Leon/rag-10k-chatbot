// API client for the FastAPI backend (proxied at /api by next.config rewrites).

export type Provider = { name: string; models: string[]; note: string; available?: boolean };

export type Config = {
  providers: Provider[];
  all_providers: Provider[];
  embeddings: string[];
  companies: string[];
  core_companies: string[];
  personas: Record<string, string>;
  default_persona: string;
  defaults: Record<string, number | string>;
};

export type Source = { company: string; page: number | string; kind: string; snippet: string };

export type ChatBody = {
  provider: string; model: string; question: string;
  history: { role: string; content: string }[];
  temperature: number; top_p: number; max_tokens: number;
  embedding: string; search_type: string; top_k: number;
  hybrid: boolean; expand: boolean; companies: string[]; persona: string;
};

export type StreamEvent =
  | { type: "scope"; scope: string[] }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; elapsed: number; model: string; provider: string }
  | { type: "error"; message: string };

export async function getConfig(): Promise<Config> {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error("config failed");
  return r.json();
}

/** Stream a chat request, yielding parsed SSE events. */
export async function* streamChat(
  body: ChatBody,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.body) throw new Error("no stream");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        try {
          yield JSON.parse(line.slice(6)) as StreamEvent;
        } catch {
          /* ignore malformed */
        }
      }
    }
  }
}
