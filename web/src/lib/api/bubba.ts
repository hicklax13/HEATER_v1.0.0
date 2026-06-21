/** Typed client for the Bubba chat backend (/api/chat/*).
 *  Calls go through the shared apiGet/apiPost/apiPut/apiDelete, which attach the
 *  Clerk bearer and throw ApiError on non-OK (callers branch on 401 = sign in). */

import { apiDelete, apiGet, apiPost, apiPut } from "./client";

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
}

export const bubba = {
  send: (body: ChatSendBody) => apiPost<ChatSendResult>("/chat/send", body),

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
