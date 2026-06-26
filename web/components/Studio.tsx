"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  type ChatBody, type Config, type Source, type StreamEvent,
  getConfig, streamChat,
} from "@/lib/api";

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  scope?: string[];
  elapsed?: number;
  model?: string;
  error?: string;
  streaming?: boolean;
};

const STARTERS = [
  "What is the main source of revenue for Alphabet, Amazon, and Microsoft?",
  "How much cash did Amazon hold at the end of its latest fiscal year?",
  "How did Microsoft's working capital change from FY2024 to FY2025?",
  "What does Amazon's 10-K say about business risk in China and India?",
];

export default function Studio() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  // model selection
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");

  // parameters
  const [temperature, setTemperature] = useState(0.2);
  const [topP, setTopP] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [embedding, setEmbedding] = useState("");
  const [topK, setTopK] = useState(6);
  const [hybrid, setHybrid] = useState(true);
  const [expand, setExpand] = useState(true);
  const [companies, setCompanies] = useState<string[]>([]);
  const [personaKey, setPersonaKey] = useState("");
  const [persona, setPersona] = useState("");

  // conversation
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // ---- init
  useEffect(() => {
    const t = (localStorage.getItem("theme") as "light" | "dark") || "light";
    setTheme(t);
    getConfig().then((c) => {
      setCfg(c);
      const d = c.defaults;
      const prov = c.providers.find((p) => p.name === d.provider)?.name
        || c.providers[0]?.name || "";
      setProvider(prov);
      const models = c.providers.find((p) => p.name === prov)?.models || [];
      setModel(models.includes(String(d.model)) ? String(d.model) : models[0] || "");
      setEmbedding(c.embeddings.includes(String(d.embedding)) ? String(d.embedding) : c.embeddings[0] || "");
      setTemperature(Number(d.temperature ?? 0.2));
      setTopP(Number(d.top_p ?? 1));
      setMaxTokens(Number(d.max_tokens ?? 1024));
      setTopK(Number(d.top_k ?? 6));
      setHybrid(Boolean(d.hybrid ?? true));
      setExpand(Boolean(d.expand ?? true));
      setCompanies(c.core_companies);
      setPersonaKey(c.default_persona);
      setPersona(c.personas[c.default_persona] || "");
    }).catch(() => setCfg(null));
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const providerObj = cfg?.providers.find((p) => p.name === provider);

  function pickProvider(name: string) {
    setProvider(name);
    const models = cfg?.providers.find((p) => p.name === name)?.models || [];
    setModel(models[0] || "");
  }

  function toggleCompany(c: string) {
    setCompanies((prev) =>
      prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]);
  }

  async function send(q: string) {
    const question = q.trim();
    if (!question || busy || !provider || !model) return;
    setInput("");
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [...m, { role: "user", content: question },
      { role: "assistant", content: "", streaming: true }]);
    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const body: ChatBody = {
      provider, model, question, history,
      temperature, top_p: topP, max_tokens: maxTokens,
      embedding, search_type: "mmr", top_k: topK,
      hybrid, expand, companies: companies.length ? companies : (cfg?.core_companies || []),
      persona,
    };

    const patch = (fn: (m: Msg) => Msg) =>
      setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));

    try {
      for await (const ev of streamChat(body, ctrl.signal) as AsyncGenerator<StreamEvent>) {
        if (ev.type === "scope") patch((m) => ({ ...m, scope: ev.scope }));
        else if (ev.type === "sources") patch((m) => ({ ...m, sources: ev.sources }));
        else if (ev.type === "token") patch((m) => ({ ...m, content: m.content + ev.text }));
        else if (ev.type === "done") patch((m) => ({ ...m, streaming: false, elapsed: ev.elapsed, model: ev.model }));
        else if (ev.type === "error") patch((m) => ({ ...m, streaming: false, error: ev.message }));
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError"))
        patch((m) => ({ ...m, streaming: false, error: String(e) }));
    } finally {
      patch((m) => ({ ...m, streaming: false }));
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
    setBusy(false);
  }

  // ----------------------------------------------------------------- render
  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* LEFT — model selection */}
      <aside className="flex w-[264px] flex-col gap-4 overflow-y-auto border-r p-4"
        style={{ background: "var(--panel)", borderColor: "var(--border)" }}>
        <div>
          <div className="flex items-center gap-2 text-[17px] font-bold">📊 10-K RAG Studio</div>
          <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
            Chat with the latest 10-K filings
          </div>
        </div>

        <Section title="Provider">
          <select className="select" value={provider} onChange={(e) => pickProvider(e.target.value)}>
            {cfg?.providers.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
          </select>
        </Section>

        <Section title="Model">
          <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
            {providerObj?.models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          {providerObj?.note && (
            <div className="mt-1.5 text-[11px] leading-snug" style={{ color: "var(--muted)" }}>
              {providerObj.note}
            </div>
          )}
        </Section>

        <Section title="Companies in scope">
          <div className="flex flex-col gap-1.5">
            {cfg?.companies.map((c) => (
              <label key={c} className="flex cursor-pointer items-center gap-2 text-[13px]">
                <input type="checkbox" checked={companies.includes(c)}
                  onChange={() => toggleCompany(c)} style={{ accentColor: "var(--accent)" }} />
                {c}
              </label>
            ))}
          </div>
        </Section>

        <div className="mt-auto flex flex-col gap-2 pt-2">
          {cfg && (
            <div className="text-[11px] leading-snug" style={{ color: "var(--muted)" }}>
              🟢 {cfg.providers.length} providers connected
            </div>
          )}
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => setMessages([])}>🗑 New chat</button>
            <button className="btn" title="Toggle theme"
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
              {theme === "light" ? "🌙" : "☀️"}
            </button>
          </div>
        </div>
      </aside>

      {/* CENTER — chat */}
      <main className="flex flex-1 flex-col" style={{ background: "var(--bg)" }}>
        <header className="flex items-center justify-between border-b px-6 py-3"
          style={{ borderColor: "var(--border)" }}>
          <div>
            <div className="text-[15px] font-bold">📊 10-K RAG Studio</div>
            <div className="mt-0.5 text-[11px]" style={{ color: "var(--muted)" }}>
              {provider} · <b>{model}</b> · emb {embedding} · k {topK}
              {expand && " · expansion"} {hybrid && " · hybrid"}
            </div>
          </div>
          {busy && <button className="btn" onClick={stop}>■ Stop</button>}
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {messages.length === 0 && (
              <div className="mt-6">
                <div className="mb-3 text-sm font-semibold">💡 Try one of these</div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {STARTERS.map((s) => (
                    <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => <Message key={i} msg={m} />)}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t px-6 py-4" style={{ borderColor: "var(--border)" }}>
          <div className="mx-auto flex max-w-3xl items-end gap-2">
            <textarea className="textarea flex-1 resize-none" rows={1} value={input}
              placeholder="Ask about the companies' 10-K filings…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
              }} />
            <button className="btn-accent btn" disabled={busy || !input.trim()}
              onClick={() => send(input)}>↑</button>
          </div>
        </div>
      </main>

      {/* RIGHT — parameter fine-tuning */}
      <aside className="flex w-[292px] flex-col gap-4 overflow-y-auto border-l p-4"
        style={{ background: "var(--panel)", borderColor: "var(--border)" }}>
        <div className="text-[13px] font-bold">🎛️ Fine-tuning</div>

        <Slider label="Temperature" value={temperature} min={0} max={1.5} step={0.05}
          onChange={setTemperature} hint="Lower = more factual" />
        <Slider label="Top-p" value={topP} min={0.1} max={1} step={0.05} onChange={setTopP} />
        <Slider label="Max tokens" value={maxTokens} min={256} max={4096} step={128}
          onChange={setMaxTokens} />

        <hr style={{ borderColor: "var(--border)" }} />

        <Section title="Embedding model">
          <select className="select" value={embedding} onChange={(e) => setEmbedding(e.target.value)}>
            {cfg?.embeddings.map((e) => <option key={e} value={e}>{e}</option>)}
          </select>
        </Section>

        <Slider label="Top-k (chunks)" value={topK} min={1} max={12} step={1} onChange={setTopK} />
        <Toggle label="Hybrid (BM25 + vector)" checked={hybrid} onChange={setHybrid} />
        <Toggle label="Query expansion" checked={expand} onChange={setExpand} />

        <hr style={{ borderColor: "var(--border)" }} />

        <Section title="Persona / system prompt">
          <select className="select" value={personaKey}
            onChange={(e) => { setPersonaKey(e.target.value); setPersona(cfg?.personas[e.target.value] || ""); }}>
            {cfg && Object.keys(cfg.personas).map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
          <textarea className="textarea mt-2" rows={5} value={persona}
            onChange={(e) => setPersona(e.target.value)} />
        </Section>
      </aside>
    </div>
  );
}

/* ----------------------------------------------------------------- atoms */
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="field-label">{title}</span>
      {children}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, hint }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; hint?: string;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="field-label">{label}</span>
        <span className="text-xs tabular-nums" style={{ color: "var(--muted)" }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <div className="mt-1 text-[10px]" style={{ color: "var(--muted)" }}>{hint}</div>}
    </div>
  );
}

function Toggle({ label, checked, onChange }: {
  label: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-2">
      <span className="field-label">{label}</span>
      <span className="switch">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="track" />
      </span>
    </label>
  );
}

function RichText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return <>{parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**")
      ? <strong key={i}>{p.slice(2, -2)}</strong>
      : <span key={i}>{p}</span>)}</>;
}

function Message({ msg }: { msg: Msg }) {
  const isUser = msg.role === "user";
  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 flex-none items-center justify-center rounded-lg text-[15px]"
        style={{ background: isUser ? "var(--accent-soft)" : "var(--panel-2)" }}>
        {isUser ? "🧑" : "🤖"}
      </div>
      <div className="min-w-0 flex-1">
        {!isUser && msg.scope && msg.scope.length > 0 && (
          <div className="mb-1 text-[11px]" style={{ color: "var(--muted)" }}>
            🔎 scoped to: {msg.scope.join(", ")}
          </div>
        )}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <details className="mb-2 rounded-lg border"
            style={{ borderColor: "var(--border)", background: "var(--elev)" }}>
            <summary className="cursor-pointer px-3 py-2 text-[12px] font-medium">
              📄 Retrieved sources ({msg.sources.length})
            </summary>
            <div className="flex flex-col gap-2 px-3 pb-3">
              {msg.sources.map((s, i) => (
                <div key={i} className="text-[11px]">
                  <span className="tag">{s.kind}</span>{" "}
                  <b>{s.company} p.{s.page}</b>
                  <div className="mt-0.5" style={{ color: "var(--muted)" }}>{s.snippet}</div>
                </div>
              ))}
            </div>
          </details>
        )}
        <div className="rounded-xl px-3.5 py-2.5"
          style={{ background: isUser ? "var(--accent-soft)" : "var(--panel)", border: "1px solid var(--border)" }}>
          <div className={"msg-body" + (msg.streaming && !msg.content ? " cursor-blink" : "")}>
            <RichText text={msg.content} />
            {msg.streaming && msg.content && <span className="cursor-blink" />}
          </div>
          {msg.error && <div className="text-[12px]" style={{ color: "var(--warn)" }}>⚠ {msg.error}</div>}
        </div>
        {!isUser && msg.elapsed != null && (
          <div className="mt-1 text-[10px]" style={{ color: "var(--muted)" }}>
            ⏱ {msg.elapsed}s · {msg.model}
          </div>
        )}
      </div>
    </div>
  );
}
