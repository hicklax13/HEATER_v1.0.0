/** Typed client for the Bubba chat backend (/api/chat/*).
 *  Calls go through the shared apiGet/apiPost/apiPut/apiDelete, which attach the
 *  Clerk bearer and throw ApiError on non-OK (callers branch on 401 = sign in). */

import { apiDelete, apiGet, apiPost, apiPut, authToken } from "./client";
import { ApiError } from "./errors";

export interface ChatMessage {
  role: string;
  content: string;
  model?: string | null;
  created_at?: string | null;
}

export interface ChatSendResult {
  content: string;
  conversation_id: number | null;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  tool_trace: unknown[];
  /** Non-null = a graceful provider/key/over-cap message (HTTP was still 200). */
  error: string | null;
}

export interface ConversationSummary {
  id: number;
  title: string;
  model?: string | null;
  updated_at?: string | null;
}

export interface ChatModelOption {
  id: string;
  label: string;
  provider: string;
}

export interface KeyMeta {
  provider: string;
  label?: string | null;
  created_at?: string | null;
}

export interface ChatSendBody {
  message: string;
  model: string;
  conversation_id?: number | null;
  web_search?: boolean;
  deep_research?: boolean;
  reasoning_effort?: "off" | "low" | "medium" | "high";
  attached_text?: string;
  attachments?: { kind: "image"; data_url: string }[];
}

/** SSE events from POST /api/chat/send-stream (mirrors src/ai/providers._chat_events). */
export type BubbaStreamEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_started"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; ok: boolean }
  | {
      type: "done";
      content: string;
      conversation_id: number | null;
      cost_usd: number;
      tokens_in: number;
      tokens_out: number;
      tool_trace: unknown[];
    }
  | { type: "error"; message: string };

/** POST the chat and invoke onEvent() per SSE frame. Mirrors apiPost's auth
 *  header (Clerk bearer). Throws ApiError on a non-OK status (401 -> sign in). */
async function sendStream(body: ChatSendBody, onEvent: (e: BubbaStreamEvent) => void): Promise<void> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  const token = await authToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch("/api/chat/send-stream", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, "/chat/send-stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)) as BubbaStreamEvent);
      } catch {
        // ignore a malformed frame rather than killing the stream
      }
    }
  }
}

export const bubba = {
  send: (body: ChatSendBody) => apiPost<ChatSendResult>("/chat/send", body),

  sendStream,

  conversations: () =>
    apiGet<{ conversations: ConversationSummary[] }>("/chat/conversations").then((r) => r.conversations),

  messages: (conversationId: number) =>
    apiGet<{ conversation_id: number; messages: ChatMessage[] }>(
      `/chat/conversations/${conversationId}/messages`,
    ).then((r) => r.messages),

  models: () => apiGet<{ models: ChatModelOption[] }>("/chat/models").then((r) => r.models),

  listKeys: () => apiGet<{ keys: KeyMeta[] }>("/chat/keys").then((r) => r.keys),

  putKey: (body: { provider: string; api_key: string; label?: string | null }) =>
    apiPut<{ ok: boolean; message: string }>("/chat/keys", body),

  deleteKey: (provider: string, label?: string | null) =>
    apiDelete<{ ok: boolean; message: string }>(
      "/chat/keys",
      label ? { provider, label } : { provider },
    ),
};
