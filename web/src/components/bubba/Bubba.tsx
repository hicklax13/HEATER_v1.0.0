"use client";

/**
 * Bubba — HEATER's AI assistant (Phase B1 chat MVP).
 *
 * A floating "Ask Bubba" launcher + a non-modal chat panel, on every page. Wraps
 * the /api/chat/* backend (the unchanged src/ai engine) via the typed `bubba`
 * client. Per-user, BYO-keys, non-streamed. READ + ANALYZE only — Bubba
 * recommends, it never makes league moves.
 *
 * Auth: chat is per-user, so the backend 401s without a Clerk session. The panel
 * handles that gracefully (a "sign in" state) rather than erroring. Rich features
 * (select-to-tag, web search, screenshots, streaming, saved prompts) + managed
 * tiers are later phases (B2/B3).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Flame, KeyRound, Loader2, Plus, Send, Settings, Trash2, X } from "lucide-react";
import { bubba, type ChatModelOption, type ConversationSummary, type KeyMeta } from "@/lib/api/bubba";
import { isAuthRequired } from "@/lib/api/errors";
import { cn } from "@/lib/utils";

type Phase = "loading" | "ready" | "auth" | "error";
interface UiMessage {
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
}

const PROVIDERS = ["deepseek", "anthropic", "openai", "gemini", "xai", "openrouter", "ollama"];

export function Bubba() {
  const [open, setOpen] = useState(false);
  const reduce = useReducedMotion();

  return (
    <>
      {/* Floating launcher — hidden while the panel is open */}
      <AnimatePresence>
        {!open && (
          <motion.button
            type="button"
            onClick={() => setOpen(true)}
            aria-label="Ask Bubba — open the AI assistant"
            initial={reduce ? false : { scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={reduce ? undefined : { scale: 0.6, opacity: 0 }}
            whileHover={reduce ? undefined : { scale: 1.05 }}
            className="fixed bottom-5 right-5 z-[60] flex items-center gap-2 rounded-full bg-gradient-to-b from-heat-bright to-heat px-4 py-3 font-display text-sm font-bold text-white shadow-[0_8px_24px_rgba(255,92,16,0.45)] ring-1 ring-white/20"
          >
            <Flame className="size-5" aria-hidden />
            Ask Bubba
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence>{open && <BubbaPanel onClose={() => setOpen(false)} />}</AnimatePresence>
    </>
  );
}

function BubbaPanel({ onClose }: { onClose: () => void }) {
  const reduce = useReducedMotion();
  const [phase, setPhase] = useState<Phase>("loading");
  const [models, setModels] = useState<ChatModelOption[]>([]);
  const [model, setModel] = useState<string>("");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const [epoch, setEpoch] = useState(0);
  const reload = useCallback(() => setEpoch((n) => n + 1), []);

  useEffect(() => {
    let alive = true;
    // All setState runs inside async .then callbacks so the synchronous effect
    // body never calls setState (satisfies react-hooks/set-state-in-effect).
    Promise.resolve()
      .then(() => {
        if (!alive) return undefined;
        setPhase("loading");
        return Promise.all([bubba.models(), bubba.conversations()]);
      })
      .then((res) => {
        if (!alive || !res) return;
        const [ms, convos] = res;
        setModels(ms);
        setModel((m) => m || ms[0]?.id || "");
        setConversations(convos);
        setPhase("ready");
      })
      .catch((e) => {
        if (alive) setPhase(isAuthRequired(e) ? "auth" : "error");
      });
    return () => {
      alive = false;
    };
  }, [epoch]);

  // Keep the transcript pinned to the latest message.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, sending]);

  const selectConversation = useCallback(async (id: number) => {
    setConversationId(id);
    try {
      const msgs = await bubba.messages(id);
      setMessages(msgs.map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content })));
    } catch {
      setMessages([]);
    }
  }, []);

  const startNew = useCallback(() => {
    setConversationId(null);
    setMessages([]);
  }, []);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !model) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setSending(true);
    try {
      const res = await bubba.send({ message: text, model, conversation_id: conversationId });
      if (res.conversation_id) setConversationId(res.conversation_id);
      setMessages((prev) => [
        ...prev,
        res.error
          ? { role: "assistant", content: res.error, isError: true }
          : { role: "assistant", content: res.content },
      ]);
    } catch (e) {
      if (isAuthRequired(e)) {
        setPhase("auth");
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "Bubba couldn't reach the server. Try again.", isError: true },
        ]);
      }
    } finally {
      setSending(false);
    }
  }, [input, sending, model, conversationId]);

  return (
    <motion.div
      role="dialog"
      aria-label="Bubba AI assistant"
      initial={reduce ? false : { y: 24, opacity: 0, scale: 0.98 }}
      animate={{ y: 0, opacity: 1, scale: 1 }}
      exit={reduce ? undefined : { y: 24, opacity: 0, scale: 0.98 }}
      transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 34 }}
      className="fixed bottom-5 right-5 z-[60] flex h-[min(560px,82vh)] w-[min(380px,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-2xl border border-line bg-canvas shadow-[0_18px_50px_rgba(11,24,48,0.30),0_0_0_1.5px_rgba(255,92,16,0.30)]"
    >
      {/* header */}
      <div className="flex shrink-0 items-center justify-between bg-navy px-4 py-3 text-white">
        <div className="flex items-center gap-2 font-display font-bold tracking-wide">
          <Flame className="size-5 text-heat" aria-hidden />
          <span>
            Ask <span className="text-heat">Bubba</span>
          </span>
        </div>
        <div className="flex items-center gap-1">
          <IconBtn label="New conversation" onClick={startNew}>
            <Plus className="size-4" aria-hidden />
          </IconBtn>
          <IconBtn label="API key settings" onClick={() => setShowSettings((s) => !s)} active={showSettings}>
            <Settings className="size-4" aria-hidden />
          </IconBtn>
          <IconBtn label="Close" onClick={onClose}>
            <X className="size-4" aria-hidden />
          </IconBtn>
        </div>
      </div>

      {phase === "loading" && <Centered><Loader2 className="size-6 animate-spin text-heat" aria-hidden /></Centered>}

      {phase === "auth" && (
        <Centered>
          <p className="max-w-[260px] text-center text-sm text-ink-2">
            Sign in to chat with Bubba — your conversations and API keys are saved to your account.
          </p>
        </Centered>
      )}

      {phase === "error" && (
        <Centered>
          <p className="text-center text-sm text-ink-2">Couldn&apos;t reach Bubba.</p>
          <button
            onClick={reload}
            className="rounded-lg bg-heat px-3 py-1.5 text-sm font-semibold text-white hover:bg-heat-bright"
          >
            Retry
          </button>
        </Centered>
      )}

      {phase === "ready" && (
        <>
          {showSettings && <KeySettings onChanged={reload} />}

          {/* controls: model + history */}
          <div className="flex shrink-0 items-center gap-2 border-b border-line bg-surface px-3 py-2">
            <select
              aria-label="Model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="min-w-0 flex-1 rounded-lg border border-line bg-canvas px-2 py-1.5 text-xs font-semibold text-ink outline-none focus:border-heat"
            >
              {models.length === 0 && <option value="">No models — add a key</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <select
              aria-label="Conversation history"
              value={conversationId ?? ""}
              onChange={(e) => (e.target.value ? selectConversation(Number(e.target.value)) : startNew())}
              className="min-w-0 max-w-[44%] rounded-lg border border-line bg-canvas px-2 py-1.5 text-xs font-semibold text-ink outline-none focus:border-heat"
            >
              <option value="">New chat</option>
              {conversations.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.title}
                </option>
              ))}
            </select>
          </div>

          {/* transcript */}
          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3 py-3">
            {messages.length === 0 && !sending && (
              <p className="mt-6 text-center text-sm text-ink-3">
                {models.length === 0
                  ? "Add your own provider API key (gear icon) to start chatting."
                  : "Ask Bubba anything about your league, players, or matchups."}
              </p>
            )}
            {messages.map((m, i) => (
              <Bubble key={i} message={m} />
            ))}
            {sending && (
              <div className="flex items-center gap-2 text-sm text-ink-3">
                <Loader2 className="size-4 animate-spin text-heat" aria-hidden /> Bubba is thinking…
              </div>
            )}
          </div>

          {/* composer */}
          <div className="shrink-0 border-t border-line bg-surface p-2">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                rows={1}
                placeholder="Ask Bubba…"
                className="max-h-28 min-h-[38px] flex-1 resize-none rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink outline-none focus:border-heat"
              />
              <button
                onClick={handleSend}
                disabled={sending || !input.trim() || !model}
                aria-label="Send"
                className="flex size-[38px] shrink-0 items-center justify-center rounded-lg bg-heat text-white transition-colors hover:bg-heat-bright disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send className="size-4" aria-hidden />
              </button>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
}

function Bubble({ message }: { message: UiMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm",
          isUser
            ? "bg-heat text-white"
            : message.isError
              ? "border border-ember/30 bg-ember/10 text-ember"
              : "bg-surface-2 text-ink",
        )}
      >
        {message.content}
      </div>
    </div>
  );
}

function KeySettings({ onChanged }: { onChanged: () => void }) {
  const [keys, setKeys] = useState<KeyMeta[]>([]);
  const [provider, setProvider] = useState(PROVIDERS[0]);
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const [epoch, setEpoch] = useState(0);
  const refresh = () => setEpoch((n) => n + 1);

  useEffect(() => {
    let alive = true;
    Promise.resolve()
      .then(() => bubba.listKeys())
      .then((ks) => {
        if (alive) setKeys(ks);
      })
      .catch(() => {
        if (alive) setKeys([]);
      });
    return () => {
      alive = false;
    };
  }, [epoch]);

  const save = async () => {
    if (!apiKey.trim() || busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await bubba.putKey({ provider, api_key: apiKey.trim() });
      setMsg(r.message);
      if (r.ok) {
        setApiKey("");
        refresh();
        onChanged();
      }
    } catch {
      setMsg("Couldn't save the key.");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (p: string, label?: string | null) => {
    await bubba.deleteKey(p, label).catch(() => undefined);
    refresh();
    onChanged();
  };

  return (
    <div className="shrink-0 space-y-2 border-b border-line bg-surface-2 px-3 py-3">
      <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-ink-3">
        <KeyRound className="size-3.5" aria-hidden /> Your API keys
      </div>
      <p className="text-[11px] text-ink-3">Add your own provider key — free, and you skip shared limits.</p>
      <div className="flex items-center gap-2">
        <select
          aria-label="Provider"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          className="rounded-lg border border-line bg-canvas px-2 py-1.5 text-xs font-semibold text-ink outline-none focus:border-heat"
        >
          {PROVIDERS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="API key"
          className="min-w-0 flex-1 rounded-lg border border-line bg-canvas px-2 py-1.5 text-xs text-ink outline-none focus:border-heat"
        />
        <button
          onClick={save}
          disabled={busy || !apiKey.trim()}
          className="rounded-lg bg-heat px-2.5 py-1.5 text-xs font-semibold text-white hover:bg-heat-bright disabled:opacity-40"
        >
          Save
        </button>
      </div>
      {msg && <p className="text-[11px] text-ink-2">{msg}</p>}
      {keys.length > 0 && (
        <ul className="space-y-1">
          {keys.map((k, i) => (
            <li key={i} className="flex items-center justify-between text-xs text-ink">
              <span>
                {k.provider}
                {k.label ? ` · ${k.label}` : ""}
              </span>
              <button
                onClick={() => remove(k.provider, k.label)}
                aria-label={`Remove ${k.provider} key`}
                className="text-ink-3 hover:text-ember"
              >
                <Trash2 className="size-3.5" aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function IconBtn({
  children,
  label,
  onClick,
  active,
}: {
  children: React.ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "flex size-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/15 hover:text-white",
        active && "bg-white/15 text-white",
      )}
    >
      {children}
    </button>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6">{children}</div>;
}
