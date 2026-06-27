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
import {
  BookMarked,
  Camera,
  FileText,
  Flame,
  Image as ImageIcon,
  KeyRound,
  Loader2,
  MonitorUp,
  MousePointerClick,
  Paperclip,
  Pencil,
  Plus,
  Send,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { bubba, type ChatModelOption, type ConversationSummary, type KeyMeta, type SavedPrompt } from "@/lib/api/bubba";
import { captureScreen, extractPdfText, readImageFile } from "./attachments";
import { useBubbaContext } from "./BubbaContext";
import { isAuthRequired } from "@/lib/api/errors";
import { cn } from "@/lib/utils";

type Phase = "loading" | "ready" | "auth" | "error";
type Effort = "off" | "low" | "medium" | "high";
interface UiMessage {
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
  nudge?: "over_cap";
}

const PROVIDERS = ["deepseek", "anthropic", "openai", "gemini", "xai", "openrouter", "ollama"];

const PAGE_CONTEXT_CAP = 16 * 1024; // 16 KB cap on the auto-attached page snapshot

/** Serialize the current page's loaded data into a size-capped page_context.
 *  Returns undefined when there's nothing to attach or the data can't be
 *  serialized (cycles / non-serializable) — never throws. The backend treats an
 *  empty/truncated data_json gracefully (the user turn stays byte-identical when
 *  absent), and the prepended block tells Bubba the data may be truncated. */
function buildPageContext(pageId: string, data: unknown): { page: string; data_json: string } | undefined {
  if (!pageId || data == null) return undefined;
  let json: string;
  try {
    json = JSON.stringify(data);
  } catch {
    return undefined; // non-serializable (cycles) — skip rather than throw
  }
  if (!json) return undefined;
  const data_json = json.length > PAGE_CONTEXT_CAP ? json.slice(0, PAGE_CONTEXT_CAP) + "…[truncated]" : json;
  return { page: pageId, data_json };
}

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
  const [effort, setEffort] = useState<Effort>("off");
  const [webSearch, setWebSearch] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  const [tags, setTags] = useState<{ id: string; kind: "text" | "player"; label: string; text: string }[]>([]);
  const [images, setImages] = useState<
    { id: string; label: string; dataUrl: string; source: "snapshot" | "screen" | "file" }[]
  >([]);
  const [docs, setDocs] = useState<{ id: string; label: string; text: string; truncated: boolean }[]>([]);
  const [attachError, setAttachError] = useState<string | null>(null);
  const [queue, setQueue] = useState<{ id: string; text: string }[]>([]);
  const [showPrompts, setShowPrompts] = useState(false);
  // Auto page-context: Bubba reads what the current page published (via usePageData)
  // and attaches a size-capped JSON of it on every message. Default ON (quiet toggle).
  const { pageId, data: pageData } = useBubbaContext();
  const [pageContextOn, setPageContextOn] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canShareScreen = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getDisplayMedia;

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

  // Select-mode: highlighting page text (outside the panel) tags it.
  useEffect(() => {
    if (!selectMode) return undefined;
    const onMouseUp = () => {
      const sel = window.getSelection();
      const text = sel?.toString().trim();
      if (!text) return;
      const anchor = sel?.anchorNode ?? null;
      if (anchor && panelRef.current?.contains(anchor)) return; // ignore in-panel selections
      const label = text.length > 48 ? text.slice(0, 48) + "…" : text;
      setTags((t) => [...t, { id: crypto.randomUUID(), kind: "text", label, text }]);
      sel?.removeAllRanges();
    };
    document.addEventListener("mouseup", onMouseUp);
    return () => document.removeEventListener("mouseup", onMouseUp);
  }, [selectMode]);

  // Select-mode: clicking a player row/card (any element carrying data-bubba-tag)
  // tags that player INSTEAD of opening its dialog. Capture phase + stopPropagation
  // intercept the click before the PlayerDialog trigger sees it.
  useEffect(() => {
    if (!selectMode) return undefined;
    const onClick = (e: MouseEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.("[data-bubba-tag]");
      if (!el || panelRef.current?.contains(el)) return;
      e.preventDefault();
      e.stopPropagation();
      const raw = el.getAttribute("data-bubba-tag") ?? "";
      const [namePart, mlbId] = raw.split("|");
      const name = namePart?.trim() || el.textContent?.trim() || ""; // fall back to element text
      if (!name) return; // nothing usable — don't add an empty chip
      const text = mlbId ? `Player: ${name} (mlbId ${mlbId})` : `Player: ${name}`;
      setTags((t) => [...t, { id: crypto.randomUUID(), kind: "player", label: name, text }]);
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, [selectMode]);

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

  const snapPage = useCallback(async () => {
    setAttachError(null);
    try {
      const { toPng } = await import("html-to-image");
      const node = (document.querySelector("main") as HTMLElement | null) ?? document.body;
      const width = node.offsetWidth || 1280;
      const pixelRatio = Math.min(1, 1280 / width); // cap output width ~1280px
      const dataUrl = await toPng(node, {
        pixelRatio,
        cacheBust: true,
        // exclude the Bubba panel + the select-mode glow from the snapshot
        filter: (n) => !(n instanceof HTMLElement && n.dataset?.bubbaPanel === "1"),
      });
      setImages((s) => [...s, { id: crypto.randomUUID(), label: "page snapshot", dataUrl, source: "snapshot" }]);
    } catch (err) {
      // html-to-image throws on a tainted canvas (OOM, security). Log the cause
      // (the file's discipline) — a bare message would hide a systemic failure.
      console.error("Bubba snapPage failed", err);
      setAttachError("Couldn't snapshot the page.");
    }
  }, []);

  const onPickFiles = useCallback(async (files: FileList | null) => {
    if (!files?.length) return;
    setAttachError(null);
    for (const file of Array.from(files)) {
      try {
        if (file.size > 8 * 1024 * 1024) {
          setAttachError(`"${file.name}" is too large (max 8 MB).`);
          continue;
        }
        if (file.type.startsWith("image/")) {
          const dataUrl = await readImageFile(file);
          setImages((s) => [...s, { id: crypto.randomUUID(), label: file.name, dataUrl, source: "file" }]);
        } else if (file.type === "application/pdf") {
          const { text, truncated } = await extractPdfText(file);
          if (!text) {
            setAttachError(`No readable text found in "${file.name}".`);
            continue;
          }
          setDocs((s) => [...s, { id: crypto.randomUUID(), label: file.name, text, truncated }]);
        } else {
          setAttachError(`"${file.name}" isn't a supported file type.`);
        }
      } catch (err) {
        console.error("Bubba file attach failed", err);
        setAttachError(`Couldn't read "${file.name}".`);
      }
    }
  }, []);

  const onShareScreen = useCallback(async () => {
    setAttachError(null);
    try {
      const dataUrl = await captureScreen();
      setImages((s) => [...s, { id: crypto.randomUUID(), label: "screen capture", dataUrl, source: "screen" }]);
    } catch (err) {
      // getDisplayMedia rejects on cancel/denial/unsupported — surface, don't throw.
      console.error("Bubba screen capture failed", err);
      setAttachError("Screen capture was cancelled or isn't available.");
    }
  }, []);

  const handleSend = useCallback(async (textArg?: string) => {
    const isQueued = textArg !== undefined;
    const text = (textArg ?? input).trim();
    if (!text || sending || !model) return;
    if (!isQueued) setInput("");
    // Append the user turn + an empty assistant turn we stream INTO.
    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setSending(true);
    setToolStatus(null);

    const appendToLast = (chunk: string) =>
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === "assistant") next[next.length - 1] = { ...last, content: last.content + chunk };
        return next;
      });

    const sentTags = isQueued ? [] : tags;
    const sentImages = isQueued ? [] : images;
    const sentDocs = isQueued ? [] : docs;
    if (!isQueued) {
      setTags([]);
      setImages([]);
      setDocs([]);
    }

    const attachedTextParts = [
      ...sentTags.map((t) => t.text),
      ...sentDocs.map((d) => `[Document: ${d.label}]\n${d.text}`),
    ];

    // Auto page-context (distinct from the manual attached_text tag flow above):
    // attached on EVERY message (incl. queued) since it reflects the live page.
    const pageCtx = pageContextOn ? buildPageContext(pageId, pageData) : undefined;

    try {
      await bubba.sendStream(
        {
          message: text,
          model,
          conversation_id: conversationId,
          web_search: webSearch,
          deep_research: deepResearch,
          reasoning_effort: effort,
          attached_text: attachedTextParts.length ? attachedTextParts.join("\n") : undefined,
          attachments: sentImages.length
            ? sentImages.map((im) => ({ kind: "image" as const, data_url: im.dataUrl }))
            : undefined,
          page_context: pageCtx,
        },
        (e) => {
          if (e.type === "text_delta") {
            appendToLast(e.text);
          } else if (e.type === "tool_started") {
            setToolStatus(toolLabel(e.name));
          } else if (e.type === "tool_result") {
            setToolStatus(null);
          } else if (e.type === "done") {
            if (e.conversation_id) setConversationId(e.conversation_id);
            setToolStatus(null);
            // An empty completion (model returned no text) would leave a blank
            // bubble that looks like a UI bug — surface it instead of nothing.
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === "assistant" && !last.content.trim()) {
                next[next.length - 1] = {
                  role: "assistant",
                  content: "Bubba returned an empty response — try rephrasing.",
                  isError: true,
                };
              }
              return next;
            });
          } else if (e.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = {
                role: "assistant",
                content: e.message,
                isError: true,
                ...(e.code === "over_cap" ? { nudge: "over_cap" as const } : {}),
              };
              return next;
            });
          }
        },
      );
    } catch (err) {
      if (isAuthRequired(err)) {
        setPhase("auth");
      } else {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: "Bubba couldn't reach the server. Try again.",
            isError: true,
          };
          return next;
        });
      }
    } finally {
      setSending(false);
      setToolStatus(null);
    }
  }, [input, sending, model, conversationId, webSearch, deepResearch, effort, tags, images, docs, pageContextOn, pageId, pageData]);

  const onSubmit = useCallback(() => {
    const text = input.trim();
    if (!text || !model) return;
    if (sending) {
      // Bubba is busy — queue this message; it auto-sends after the current turn.
      setQueue((q) => [...q, { id: crypto.randomUUID(), text }]);
      setInput("");
    } else {
      handleSend();
    }
  }, [input, model, sending, handleSend]);

  // Drain the queue: when Bubba finishes a turn (sending→false) cleanly, auto-send
  // the next queued message. Pause on an errored turn so it doesn't blast the rest.
  // setState is deferred to a microtask (the file's pattern) to satisfy
  // react-hooks/set-state-in-effect; removed by id so a concurrent edit is safe.
  useEffect(() => {
    if (sending || queue.length === 0) return undefined;
    const last = messages[messages.length - 1];
    if (last?.isError) return undefined;
    const next = queue[0];
    let alive = true;
    Promise.resolve().then(() => {
      if (!alive) return;
      setQueue((q) => q.filter((x) => x.id !== next.id));
      handleSend(next.text);
    });
    return () => {
      alive = false;
    };
  }, [sending, queue, messages, handleSend]);

  return (
    <>
      {selectMode && (
        <div aria-hidden className="pointer-events-none fixed inset-0 z-[55] ring-2 ring-inset ring-heat/60" />
      )}
      <motion.div
        ref={panelRef}
        data-bubba-panel="1"
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
            {messages.map((m, i) =>
              m.nudge === "over_cap" ? (
                <OverCapNudge key={i} message={m.content} onAddKey={() => setShowSettings(true)} />
              ) : (
                <Bubble key={i} message={m} />
              ),
            )}
            {sending && (
              <div className="flex items-center gap-2 text-sm text-ink-3">
                <Loader2 className="size-4 animate-spin text-heat" aria-hidden />
                {toolStatus ?? "Bubba is thinking…"}
              </div>
            )}
          </div>

          {/* composer */}
          <div className="shrink-0 border-t border-line bg-surface p-2">
            {showPrompts && (
              <PromptsMenu
                currentInput={input}
                onPick={(t) => {
                  setInput(t);
                  setShowPrompts(false);
                }}
              />
            )}
            {queue.length > 0 && !sending && messages[messages.length - 1]?.isError && (
              <p className="mb-1 text-[11px] text-ember">Queue paused after an error — send a message to resume.</p>
            )}
            {queue.length > 0 && (
              <div className="mb-2 space-y-1">
                {queue.map((q) => (
                  <div
                    key={q.id}
                    className="flex items-center gap-1 rounded-lg border border-line bg-canvas px-2 py-1 text-[11px] text-ink"
                  >
                    <span className="flex-1 truncate">{q.text}</span>
                    <button
                      type="button"
                      onClick={() => {
                        setInput(q.text);
                        setQueue((x) => x.filter((y) => y.id !== q.id));
                      }}
                      aria-label="Edit queued message"
                      className="text-ink-3 hover:text-heat"
                    >
                      <Pencil className="size-3" aria-hidden />
                    </button>
                    <button
                      type="button"
                      onClick={() => setQueue((x) => x.filter((y) => y.id !== q.id))}
                      aria-label="Delete queued message"
                      className="text-ink-3 hover:text-ember"
                    >
                      <X className="size-3" aria-hidden />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {(tags.length > 0 || images.length > 0 || docs.length > 0) && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {tags.map((t) => (
                  <TagChip key={t.id} label={t.label} onRemove={() => setTags((x) => x.filter((y) => y.id !== t.id))} />
                ))}
                {images.map((im) => (
                  <TagChip
                    key={im.id}
                    label={im.label}
                    icon={<ImageIcon className="size-3 shrink-0 text-heat" aria-hidden />}
                    onRemove={() => setImages((x) => x.filter((y) => y.id !== im.id))}
                  />
                ))}
                {docs.map((d) => (
                  <TagChip
                    key={d.id}
                    label={d.truncated ? `${d.label} · truncated` : d.label}
                    icon={<FileText className="size-3 shrink-0 text-heat" aria-hidden />}
                    onRemove={() => setDocs((x) => x.filter((y) => y.id !== d.id))}
                  />
                ))}
              </div>
            )}
            <div className="mb-2 flex items-center gap-2 text-[11px]">
              <select
                aria-label="Thinking effort"
                value={effort}
                onChange={(e) => setEffort(e.target.value as Effort)}
                className="rounded-lg border border-line bg-canvas px-2 py-1 font-semibold text-ink outline-none focus:border-heat"
              >
                <option value="off">Effort: Off</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
              <Toggle label="Web" on={webSearch} onClick={() => setWebSearch((v) => !v)} />
              <Toggle label="Research" on={deepResearch} onClick={() => setDeepResearch((v) => !v)} />
              <Toggle
                label="Page"
                on={pageContextOn}
                onClick={() => setPageContextOn((v) => !v)}
                title="Let Bubba see this page's data"
              />
              <button
                type="button"
                onClick={() => setSelectMode((v) => !v)}
                aria-pressed={selectMode}
                title="Tag page content — highlight text or click a player"
                className={cn(
                  "ml-auto flex items-center gap-1 rounded-full border px-2.5 py-1 font-semibold transition-colors",
                  selectMode ? "border-heat bg-heat/10 text-heat" : "border-line bg-canvas text-ink-3 hover:text-ink",
                )}
              >
                <MousePointerClick className="size-3.5" aria-hidden /> Tag
              </button>
              <button
                type="button"
                onClick={snapPage}
                title="Attach a snapshot of this page"
                className="flex items-center gap-1 rounded-full border border-line bg-canvas px-2.5 py-1 font-semibold text-ink-3 transition-colors hover:text-ink"
              >
                <Camera className="size-3.5" aria-hidden /> Snap
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="Attach an image or PDF"
                className="flex items-center gap-1 rounded-full border border-line bg-canvas px-2.5 py-1 font-semibold text-ink-3 transition-colors hover:text-ink"
              >
                <Paperclip className="size-3.5" aria-hidden /> File
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,application/pdf"
                multiple
                className="hidden"
                onChange={(e) => {
                  onPickFiles(e.target.files);
                  e.target.value = ""; // allow re-picking the same file
                }}
              />
              {canShareScreen && (
                <button
                  type="button"
                  onClick={onShareScreen}
                  title="Capture an open window or screen"
                  className="flex items-center gap-1 rounded-full border border-line bg-canvas px-2.5 py-1 font-semibold text-ink-3 transition-colors hover:text-ink"
                >
                  <MonitorUp className="size-3.5" aria-hidden /> Screen
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowPrompts((v) => !v)}
                aria-pressed={showPrompts}
                title="Saved prompts"
                className={cn(
                  "flex items-center gap-1 rounded-full border px-2.5 py-1 font-semibold transition-colors",
                  showPrompts ? "border-heat bg-heat/10 text-heat" : "border-line bg-canvas text-ink-3 hover:text-ink",
                )}
              >
                <BookMarked className="size-3.5" aria-hidden /> Prompts
              </button>
            </div>
            {attachError && <p className="mb-1 text-[11px] text-ember">{attachError}</p>}
            {images.some((im) => im.source === "snapshot") && (
              <p className="mb-1 text-[11px] text-ink-3">
                Page snapshot captures text + charts; some photos may not render.
              </p>
            )}
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
                rows={1}
                placeholder={sending ? "Queue a follow-up…" : "Ask Bubba…"}
                className="max-h-28 min-h-[38px] flex-1 resize-none rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink outline-none focus:border-heat"
              />
              <button
                onClick={onSubmit}
                disabled={!input.trim() || !model}
                aria-label={sending ? "Queue message" : "Send"}
                title={sending ? "Queue message (sends after the current answer)" : "Send"}
                className="flex size-[38px] shrink-0 items-center justify-center rounded-lg bg-heat text-white transition-colors hover:bg-heat-bright disabled:cursor-not-allowed disabled:opacity-40"
              >
                {sending ? <Plus className="size-4" aria-hidden /> : <Send className="size-4" aria-hidden />}
              </button>
            </div>
          </div>
        </>
      )}
      </motion.div>
    </>
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

function OverCapNudge({ message, onAddKey }: { message: string; onAddKey: () => void }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-2 rounded-2xl border border-heat/30 bg-heat/5 px-3 py-2 text-sm text-ink">
        <p>{message}</p>
        <div className="flex flex-wrap gap-2">
          <a
            href="/pricing"
            className="rounded-lg bg-heat px-2.5 py-1 text-xs font-semibold text-white hover:bg-heat-bright"
          >
            Upgrade
          </a>
          <button
            type="button"
            onClick={onAddKey}
            className="rounded-lg border border-line bg-canvas px-2.5 py-1 text-xs font-semibold text-ink hover:text-heat"
          >
            Add your own key
          </button>
        </div>
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

function Toggle({
  label,
  on,
  onClick,
  title,
}: {
  label: string;
  on: boolean;
  onClick: () => void;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      title={title}
      className={cn(
        "rounded-full border px-2.5 py-1 font-semibold transition-colors",
        on ? "border-heat bg-heat/10 text-heat" : "border-line bg-canvas text-ink-3 hover:text-ink",
      )}
    >
      {label}
    </button>
  );
}

function TagChip({ label, onRemove, icon }: { label: string; onRemove: () => void; icon?: React.ReactNode }) {
  return (
    <span className="inline-flex max-w-[200px] items-center gap-1 rounded-full border border-heat/40 bg-heat/5 px-2 py-0.5 text-[11px] text-ink">
      {icon}
      <span className="truncate">{label}</span>
      <button type="button" onClick={onRemove} aria-label={`Remove ${label}`} className="text-ink-3 hover:text-ember">
        <X className="size-3" aria-hidden />
      </button>
    </span>
  );
}

function PromptsMenu({ currentInput, onPick }: { currentInput: string; onPick: (text: string) => void }) {
  const [prompts, setPrompts] = useState<SavedPrompt[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    let alive = true;
    Promise.resolve()
      .then(() => bubba.savedPrompts())
      .then((ps) => {
        if (alive) setPrompts(ps);
      })
      .catch(() => {
        if (alive) setPrompts([]);
      });
    return () => {
      alive = false;
    };
  }, [epoch]);

  const save = async () => {
    const n = name.trim();
    if (!n || !currentInput.trim() || busy) return;
    setBusy(true);
    setErr(null);
    try {
      // The backend returns {ok,message} (HTTP 200 even on a validation/store
      // failure) — surface it rather than letting a failed save look like success.
      const r = await bubba.savePrompt({ name: n, text: currentInput.trim() });
      if (!r.ok) {
        setErr(r.message || "Couldn't save prompt.");
        return;
      }
      setName("");
      setEpoch((e) => e + 1);
    } catch {
      setErr("Couldn't save prompt — check your connection.");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: number) => {
    setErr(null);
    try {
      const r = await bubba.deletePrompt(id);
      if (!r.ok) setErr(r.message || "Couldn't delete prompt.");
    } catch {
      setErr("Couldn't delete prompt — check your connection.");
    }
    setEpoch((e) => e + 1);
  };

  return (
    <div className="mb-2 space-y-2 rounded-lg border border-line bg-surface-2 p-2">
      <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wide text-ink-3">
        <BookMarked className="size-3.5" aria-hidden /> Saved prompts
      </div>
      {prompts.length === 0 && <p className="text-[11px] text-ink-3">No saved prompts yet.</p>}
      {err && <p className="text-[11px] text-ember">{err}</p>}
      {prompts.map((p) => (
        <div key={p.id} className="flex items-center gap-1 text-[11px] text-ink">
          <button
            type="button"
            onClick={() => onPick(p.text)}
            className="flex-1 truncate text-left font-semibold hover:text-heat"
            title={p.text}
          >
            {p.name}
          </button>
          <button
            type="button"
            onClick={() => remove(p.id)}
            aria-label={`Delete ${p.name}`}
            className="text-ink-3 hover:text-ember"
          >
            <Trash2 className="size-3" aria-hidden />
          </button>
        </div>
      ))}
      <div className="flex items-center gap-1.5 pt-1">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name this prompt"
          className="min-w-0 flex-1 rounded-lg border border-line bg-canvas px-2 py-1 text-[11px] text-ink outline-none focus:border-heat"
        />
        <button
          type="button"
          onClick={save}
          disabled={busy || !name.trim() || !currentInput.trim()}
          className="rounded-lg bg-heat px-2 py-1 text-[11px] font-semibold text-white hover:bg-heat-bright disabled:opacity-40"
        >
          Save current
        </button>
      </div>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6">{children}</div>;
}

function toolLabel(name: string): string {
  const map: Record<string, string> = {
    query_data: "querying the database…",
    get_player: "looking up the player…",
    compare_players: "comparing players…",
    get_my_team: "reading your roster…",
    get_standings: "checking the standings…",
    get_free_agents: "scanning free agents…",
    web_search: "searching the web…",
    deep_research: "researching…",
    request_refresh: "queuing a data refresh…",
  };
  return map[name] ?? `running ${name}…`;
}
