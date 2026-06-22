# Bubba B2.3 — Attachments (Files, Screen Capture, Documents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user attach context to a Bubba question three new ways — image files, a captured screen/window, and PDFs (as extracted text) — all feeding the `attachments`/`attached_text` request fields B2.2 already shipped.

**Architecture:** Frontend-only (the `api/`/`src/ai/` backend is untouched — images are more `attachments`, PDF text is more `attached_text`). Browser helpers live in a new `web/src/components/bubba/attachments.ts` (`readImageFile`, `captureScreen`, `extractPdfText`) so `Bubba.tsx` stays focused. The B2.2 `shots` state is renamed `images` (now holding page snapshots + image files + screen captures, each tagged with a `source`), and a new `docs` state holds extracted PDF text. `pdfjs-dist` is dynamically imported so it never enters the main bundle.

**Tech Stack:** Next.js 16 + React 19 + TypeScript; browser `FileReader`, `getDisplayMedia`, `<canvas>`; `pdfjs-dist` (new, dynamic import). No backend deps, no `api/` change.

**Coordination:** Touches `web/` only. Build in three slices (A image files → B screen capture → C PDFs); each ships + verifies independently. The B2.1/B2.2 backend invariants are confirmed unchanged by re-running the chat test suite at the end.

---

## Grounding (verified in source — do NOT re-derive)

- **Backend is already done (B2.2):** `ChatSendRequest` has `attached_text`/`attachments`; `ChatService._build_user_content` wraps text + builds the multimodal image list. **No `api/` edit in this wave.**
- `web/src/components/bubba/Bubba.tsx` (post-B2.2) `BubbaPanel` state includes:
  - `tags` (`{id, kind:"text"|"player", label, text}[]`) → `attached_text`.
  - `shots` (`{id, dataUrl}[]`) → `attachments` (page snapshots from B2.2's `snapPage`).
  - `snapError` (string|null) — shown under the controls; the B2.2 fidelity caveat renders when `shots.length > 0`.
  - `handleSend` snapshots `sentTags`/`sentShots`, clears them, and sends `attached_text: sentTags.length ? sentTags.map(t=>t.text).join("\n") : undefined` + `attachments: sentShots.length ? sentShots.map(s=>({kind:"image" as const, data_url:s.dataUrl})) : undefined`.
  - Composer controls row (`<div className="mb-2 flex items-center gap-2 text-[11px]">`) holds the effort dial + Web/Research `Toggle`s + the B2.2 Tag/Snap buttons.
  - `snapPage` pushes `{ id: crypto.randomUUID(), dataUrl }` to `shots`.
  - Helpers at file bottom: `TagChip({label, onRemove, icon?: boolean})` (Camera icon when `icon`), `Toggle`, `toolLabel`, lucide imports include `Camera`, `MousePointerClick`, `X`.
- `web/src/lib/api/bubba.ts::ChatSendBody` already has `attached_text?` + `attachments?: {kind:"image"; data_url:string}[]` — no client-type change.
- `getDisplayMedia` → `MediaStream`; grab a frame by drawing the video track to a canvas (`toDataURL`); always stop tracks after.
- `pdfjs-dist` v4+ is ESM; text via `getDocument({data}).promise` → per-page `getTextContent()`; needs `GlobalWorkerOptions.workerSrc`. `web/AGENTS.md`: this Next has breaking changes — the worker wiring is verified at build time in Slice C (with a `/public` fallback).

**Frontend commands from `web/` with `pnpm`. Git from repo root `C:/Users/conno/Code/HEATER_v1.0.1`.**

---

## File Structure

- **Create** `web/src/components/bubba/attachments.ts` — `readImageFile`, `captureScreen`, `extractPdfText` (the browser-only helpers).
- **Modify** `web/src/components/bubba/Bubba.tsx` — rename `shots`→`images` (+ `source`), rename `snapError`→`attachError`, generalize `TagChip` icon, add paperclip + share-screen buttons + handlers, add `docs` state + chips, fold into send.
- **Modify** `web/package.json` / lockfile — add `pdfjs-dist` (Slice C).

No `api/`, `src/`, or test-file changes (frontend-only wave; gates are tsc + lint + build + preview + the existing chat suite).

---

## SLICE A — Image-file attach (+ the `images`/`attachError` renames)

### Task 1: `attachments.ts` with `readImageFile` + the Bubba image-file wiring

**Files:**
- Create: `web/src/components/bubba/attachments.ts`
- Modify: `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Create `attachments.ts` with `readImageFile`**

```typescript
// web/src/components/bubba/attachments.ts
// Browser-only attachment helpers for Bubba (kept out of Bubba.tsx so the
// component stays focused, and so pdfjs-dist is dynamically imported on demand).

/** Read an image File as a data: URL (for an image attachment). */
export function readImageFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error("file read failed"));
    reader.readAsDataURL(file);
  });
}
```

- [ ] **Step 2: Rename `shots`→`images` (with a `source`) and `snapError`→`attachError` in `Bubba.tsx`.**

Replace the B2.2 state declarations:
```typescript
  const [shots, setShots] = useState<{ id: string; dataUrl: string }[]>([]);
  const [snapError, setSnapError] = useState<string | null>(null);
```
with:
```typescript
  const [images, setImages] = useState<{ id: string; label: string; dataUrl: string; source: "snapshot" | "screen" | "file" }[]>([]);
  const [docs, setDocs] = useState<{ id: string; label: string; text: string }[]>([]);
  const [attachError, setAttachError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
```

- [ ] **Step 3: Update `snapPage` to push an `images` item with `source:"snapshot"`.** Replace the B2.2 body's `setShots(...)` + `catch`:
```typescript
      setImages((s) => [...s, { id: crypto.randomUUID(), label: "page snapshot", dataUrl, source: "snapshot" }]);
    } catch (err) {
      console.error("Bubba snapPage failed", err);
      setAttachError("Couldn't snapshot the page.");
    }
  }, []);
```

- [ ] **Step 4: Add the image-file picker handler** (next to `snapPage`):
```typescript
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
        } else {
          setAttachError(`"${file.name}" isn't a supported file type.`);
        }
      } catch (err) {
        console.error("Bubba file attach failed", err);
        setAttachError(`Couldn't read "${file.name}".`);
      }
    }
  }, []);
```
and add the import at the top of `Bubba.tsx`:
```typescript
import { captureScreen, extractPdfText, readImageFile } from "./attachments";
```
(`captureScreen`/`extractPdfText` land in Slices B/C; import them now and the unused-import lint is avoided because they're referenced by handlers added in this task's later steps — if lint flags an unused import in Slice A, import only `readImageFile` here and widen the import in B/C.)

> **Executor note:** to keep each slice lint-clean, import ONLY what the slice uses. In Slice A import `readImageFile`; widen to add `captureScreen` in Slice B and `extractPdfText` in Slice C.

So for Slice A, the import is:
```typescript
import { readImageFile } from "./attachments";
```

- [ ] **Step 5: Update the chips render** to use `images` + `docs` and generalize the icon. Replace the B2.2 chips block:
```tsx
            {(tags.length > 0 || shots.length > 0) && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {tags.map((t) => (
                  <TagChip key={t.id} label={t.label} onRemove={() => setTags((x) => x.filter((y) => y.id !== t.id))} />
                ))}
                {shots.map((s) => (
                  <TagChip
                    key={s.id}
                    label="page snapshot"
                    icon
                    onRemove={() => setShots((x) => x.filter((y) => y.id !== s.id))}
                  />
                ))}
              </div>
            )}
```
with:
```tsx
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
                    label={d.label}
                    icon={<FileText className="size-3 shrink-0 text-heat" aria-hidden />}
                    onRemove={() => setDocs((x) => x.filter((y) => y.id !== d.id))}
                  />
                ))}
              </div>
            )}
```

- [ ] **Step 6: Generalize `TagChip`'s `icon` prop** from `boolean` to a node. Replace:
```tsx
function TagChip({ label, onRemove, icon }: { label: string; onRemove: () => void; icon?: boolean }) {
  return (
    <span className="inline-flex max-w-[200px] items-center gap-1 rounded-full border border-heat/40 bg-heat/5 px-2 py-0.5 text-[11px] text-ink">
      {icon && <Camera className="size-3 shrink-0 text-heat" aria-hidden />}
      <span className="truncate">{label}</span>
```
with:
```tsx
function TagChip({ label, onRemove, icon }: { label: string; onRemove: () => void; icon?: React.ReactNode }) {
  return (
    <span className="inline-flex max-w-[200px] items-center gap-1 rounded-full border border-heat/40 bg-heat/5 px-2 py-0.5 text-[11px] text-ink">
      {icon}
      <span className="truncate">{label}</span>
```

- [ ] **Step 7: Fix the lucide imports** — `Camera` is no longer used by `TagChip` but IS used by the Snap button; add `ImageIcon`/`FileText`/`Paperclip`. Replace the import line:
```typescript
import { Camera, Flame, KeyRound, Loader2, MousePointerClick, Plus, Send, Settings, Trash2, X } from "lucide-react";
```
with:
```typescript
import {
  Camera,
  FileText,
  Flame,
  Image as ImageIcon,
  KeyRound,
  Loader2,
  MousePointerClick,
  Paperclip,
  Plus,
  Send,
  Settings,
  Trash2,
  X,
} from "lucide-react";
```

- [ ] **Step 8: Update the snapshot fidelity caveat** (B2.2) to key on captured snapshots only, and the `attachError` rename. Replace:
```tsx
            {snapError && <p className="mb-1 text-[11px] text-ember">{snapError}</p>}
            {shots.length > 0 && (
              <p className="mb-1 text-[11px] text-ink-3">
                Snapshot captures page text + charts; some photos may not render.
              </p>
            )}
```
with:
```tsx
            {attachError && <p className="mb-1 text-[11px] text-ember">{attachError}</p>}
            {images.some((im) => im.source === "snapshot") && (
              <p className="mb-1 text-[11px] text-ink-3">
                Page snapshot captures text + charts; some photos may not render.
              </p>
            )}
```

- [ ] **Step 9: Add the paperclip button + hidden file input** to the controls row, after the Snap button:
```tsx
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
            </div>
```
(This replaces the closing `</div>` of the controls row that immediately followed the Snap button — keep exactly one closing `</div>`.)

- [ ] **Step 10: Update `handleSend`** to send `images`/`docs` and clear all three. Replace the B2.2 capture/clear + body:
```typescript
    const sentTags = tags;
    const sentShots = shots;
    setTags([]);
    setShots([]);

    try {
      await bubba.sendStream(
        {
          message: text,
          model,
          conversation_id: conversationId,
          web_search: webSearch,
          deep_research: deepResearch,
          reasoning_effort: effort,
          attached_text: sentTags.length ? sentTags.map((t) => t.text).join("\n") : undefined,
          attachments: sentShots.length
            ? sentShots.map((s) => ({ kind: "image" as const, data_url: s.dataUrl }))
            : undefined,
        },
```
with:
```typescript
    const sentTags = tags;
    const sentImages = images;
    const sentDocs = docs;
    setTags([]);
    setImages([]);
    setDocs([]);

    const attachedTextParts = [
      ...sentTags.map((t) => t.text),
      ...sentDocs.map((d) => `[Document: ${d.label}]\n${d.text}`),
    ];

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
        },
```
and update the `useCallback` dep array — replace `..., effort, tags, shots]);` with:
```typescript
  }, [input, sending, model, conversationId, webSearch, deepResearch, effort, tags, images, docs]);
```

- [ ] **Step 11: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green. (If lint flags `captureScreen`/`extractPdfText` as unused, confirm Step 4's executor note — only `readImageFile` is imported in Slice A.)

- [ ] **Step 12: Commit**

```bash
git add web/src/components/bubba/attachments.ts web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba image-file attach + unify image attachments (B2.3 slice A)"
```

---

## SLICE B — Screen capture

### Task 2: `captureScreen` + the share-screen button

**Files:**
- Modify: `web/src/components/bubba/attachments.ts`, `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add `captureScreen` to `attachments.ts`**

```typescript
/** Capture one frame of a user-chosen screen/window as a PNG data URL. */
export async function captureScreen(): Promise<string> {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
  try {
    const video = document.createElement("video");
    video.srcObject = stream;
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve();
    });
    await video.play();
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d canvas context");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    video.pause();
    return canvas.toDataURL("image/png");
  } finally {
    stream.getTracks().forEach((t) => t.stop()); // always release the capture
  }
}
```

- [ ] **Step 2: Widen the import** in `Bubba.tsx`:
```typescript
import { captureScreen, readImageFile } from "./attachments";
```

- [ ] **Step 3: Add the `onShareScreen` handler** (next to `onPickFiles`):
```typescript
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
```

- [ ] **Step 4: Add a capability flag + the share-screen button.** Add the flag near the top of `BubbaPanel` (after the state):
```typescript
  const canShareScreen = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getDisplayMedia;
```
and add the button in the controls row, right after the File button + its `<input>`:
```tsx
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
```
and add `MonitorUp` to the lucide import (alphabetical, after `Loader2`):
```typescript
  Loader2,
  MonitorUp,
  MousePointerClick,
```

- [ ] **Step 5: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/bubba/attachments.ts web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba screen capture via getDisplayMedia (B2.3 slice B)"
```

---

## SLICE C — PDF documents (text extraction)

### Task 3: `pdfjs-dist` + `extractPdfText` + route PDFs to `docs`

**Files:**
- Modify: `web/package.json` (+ lockfile), `web/src/components/bubba/attachments.ts`, `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add the dependency**

Run (from `web/`): `pnpm add pdfjs-dist`
Expected: `pdfjs-dist` in `web/package.json` + lockfile.

- [ ] **Step 2: Add `extractPdfText` to `attachments.ts`** (dynamic import keeps pdfjs out of the main bundle):

```typescript
const PDF_TEXT_CAP = 12000;

/** Extract text from a PDF File in the browser (capped). Returns "" on no text. */
export async function extractPdfText(file: File): Promise<string> {
  const pdfjs = await import("pdfjs-dist");
  // Worker: the bundler resolves the worker asset from the package via new URL().
  pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "pdfjs-dist/build/pdf.worker.min.mjs",
    import.meta.url,
  ).href;
  const data = new Uint8Array(await file.arrayBuffer());
  const doc = await pdfjs.getDocument({ data }).promise;
  let text = "";
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const content = await page.getTextContent();
    text += content.items.map((it) => ("str" in it ? it.str : "")).join(" ") + "\n";
    if (text.length >= PDF_TEXT_CAP) break;
  }
  text = text.trim();
  return text.length > PDF_TEXT_CAP ? text.slice(0, PDF_TEXT_CAP) + "\n…(truncated)" : text;
}
```

- [ ] **Step 3: Widen the import** in `Bubba.tsx`:
```typescript
import { captureScreen, extractPdfText, readImageFile } from "./attachments";
```

- [ ] **Step 4: Route PDFs in `onPickFiles`.** Replace the `else` branch (the "isn't a supported file type" arm) so PDFs extract to `docs`:
```typescript
        if (file.type.startsWith("image/")) {
          const dataUrl = await readImageFile(file);
          setImages((s) => [...s, { id: crypto.randomUUID(), label: file.name, dataUrl, source: "file" }]);
        } else if (file.type === "application/pdf") {
          const text = await extractPdfText(file);
          if (!text) {
            setAttachError(`No readable text found in "${file.name}".`);
            continue;
          }
          setDocs((s) => [...s, { id: crypto.randomUUID(), label: file.name, text }]);
        } else {
          setAttachError(`"${file.name}" isn't a supported file type.`);
        }
```

- [ ] **Step 5: Typecheck + lint + build — and VERIFY THE PDF WORKER BUNDLES.**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

**If the build FAILS to resolve `pdf.worker.min.mjs`** (a known Next/bundler friction point per `web/AGENTS.md`), apply the `/public` static-asset fallback instead of the `new URL` worker line:
1. Copy the worker into the app's public dir:
   `cp web/node_modules/pdfjs-dist/build/pdf.worker.min.mjs web/public/pdf.worker.min.mjs`
2. In `extractPdfText`, replace the `workerSrc` line with:
   ```typescript
   pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
   ```
3. Re-run the build (now the worker is served as a static asset, bundler-agnostic) and `git add web/public/pdf.worker.min.mjs`.

- [ ] **Step 6: Commit**

```bash
git add web/package.json web/pnpm-lock.yaml web/src/components/bubba/attachments.ts web/src/components/bubba/Bubba.tsx
# include web/public/pdf.worker.min.mjs ONLY if the fallback in Step 5 was used
git commit -m "feat(web): Bubba PDF attach via client-side text extraction (B2.3 slice C)"
```

---

### Task 4: Verify, preview, review, ship

- [ ] **Step 1: Confirm the backend invariants are untouched** (no `api/` change this wave — re-run the chat + AI suite)

Run: `.venv/Scripts/python.exe -m pytest tests/api/ tests/test_ai_providers.py tests/test_ai_providers_streaming.py tests/test_ai_thinking_params.py -q`
Expected: PASS (B2.1 `chat()` equivalence + B2.2 contracts still green).

- [ ] **Step 2: Re-assert the read-only tool surface**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_tool_surface_readonly.py -q`
Expected: PASS.

- [ ] **Step 3: Preview smoke.** From `web/`, `preview_start` the `heater-web` config; confirm the page loads with no console errors, open the Bubba launcher, and verify the panel mounts cleanly (the composer is Clerk-gated → "error"/"auth" phase locally, so the File/Screen buttons live behind the ready phase; the smoke confirms no runtime mount regression from the renames). Stop the preview. (Live attachment round-trips need a Clerk session → owner verifies post-deploy, same as B2.1/B2.2.)

- [ ] **Step 4: Code review.** Dispatch `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` on `git diff master...HEAD`. Verify: the `shots`→`images` rename is complete (no dangling `shots`/`setShots`/`snapError`); per-item attach failures are logged + surfaced, never silent; `captureScreen` always stops its tracks (no dangling screen capture); `extractPdfText`'s dynamic import keeps pdfjs out of the main bundle; the send mapping folds `docs`→`attached_text` + `images`→`attachments` correctly; no write tool / no backend change. Apply findings.

- [ ] **Step 5: Reconcile + merge + push.** `git checkout master && git pull --no-rebase origin master`, then `git merge --ff-only feat/bubba-b2-3-attachments` (or `--no-ff` if master moved), confirm the chat suite green, `git push origin master`. Pre-push runs the structural suite.

- [ ] **Step 6: Docs.** Update `CLAUDE.md` (Bubba line) + the `project_bubba_ai_assistant` memory to mark B2.3 shipped (note: documents go as extracted text; native vision-PDF deferred); flip the spec `Status:` to shipped.

---

## Self-Review

**1. Spec coverage**
- Image files → Task 1 (paperclip + `readImageFile` + `images`/`source:"file"`). ✅
- Screen capture → Task 2 (`captureScreen` + share-screen button + `source:"screen"`). ✅
- PDFs as extracted text → Task 3 (`pdfjs-dist` + `extractPdfText` + `docs` → `attached_text`). ✅
- Documents universal (any model), text-extraction not native passthrough → Task 3 `extractPdfText`, folded into `attached_text` (no `attachments`/no contract change). ✅
- Backend unchanged → no `api/` task; Task 4 Step 1 re-runs the suite to prove it. ✅
- Unify image sources (`shots`→`images` + `source`) → Task 1 Steps 2/3/5/8. ✅
- Helpers factored into `attachments.ts` → Tasks 1/2/3. ✅
- pdfjs dynamic import (out of main bundle) → Task 3 Step 2. ✅
- Error handling per-item + logged → `onPickFiles`/`onShareScreen`/`extractPdfText` catches (Tasks 1/2/3). ✅
- Caveat keyed to captured snapshots only → Task 1 Step 8 (`source === "snapshot"`). ✅
- Worker-wiring risk + `/public` fallback → Task 3 Step 5. ✅
- Read-only invariant re-assert → Task 4 Step 2. ✅
- Out-of-scope (native vision-PDF, B2.4) → not in any task. ✅

**2. Placeholder scan** — every code step shows complete code. Concrete tunables flagged: 8 MB file cap, 12k-char PDF text cap, the worker-wiring fallback (both code paths shown). The Slice-A import-narrowing executor note prevents an unused-import lint across slices. No TBD/TODO.

**3. Type consistency** — `images` items are `{id,label,dataUrl,source:"snapshot"|"screen"|"file"}` everywhere (state, `snapPage`, `onPickFiles`, `onShareScreen`, chips, send map, caveat). `docs` items are `{id,label,text}` (state, `onPickFiles`, chips, send fold). `attachError`/`setAttachError` replace `snapError`/`setSnapError` at every site. `TagChip.icon` is `React.ReactNode` (passed `<ImageIcon/>`/`<FileText/>`, omitted for text tags). `attachments.ts` exports `readImageFile`/`captureScreen`/`extractPdfText` with the signatures used in `Bubba.tsx`. lucide imports (`Camera`/`FileText`/`ImageIcon`/`Paperclip`/`MonitorUp`) match every usage.
