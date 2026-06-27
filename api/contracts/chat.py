"""Bubba chat contracts (Phase B1). Request/response models for /api/chat/*."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ChatAttachment(BaseModel):
    kind: Literal["image"] = "image"
    data_url: str  # a data:image/...;base64,... URL


class PageContext(BaseModel):
    """Structured snapshot of what the page is currently displaying, attached to
    a chat turn so Bubba can 'see the screen'. `data_json` is a (size-capped,
    possibly truncated) JSON string the frontend serializes from the page's
    loaded data; `page` is the route id (e.g. 'optimizer')."""

    page: str = ""
    data_json: str = ""


class ChatSendRequest(BaseModel):
    message: str
    model: str
    conversation_id: int | None = None
    web_search: bool = False
    deep_research: bool = False
    reasoning_effort: Literal["off", "low", "medium", "high"] | None = None
    attached_text: str | None = None
    attachments: list[ChatAttachment] | None = None
    page_context: PageContext | None = None


class ToolTraceEntry(BaseModel):
    """One tool call in the AI assistant's trace (name + args)."""

    name: str = ""
    args: dict = {}


class ChatSendResponse(BaseModel):
    content: str
    conversation_id: int | None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_trace: list[ToolTraceEntry] = []
    error: str | None = None  # graceful provider/key/over-cap message; HTTP stays 200


class ConversationSummary(BaseModel):
    id: int
    title: str
    model: str | None = None
    updated_at: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = []


class ChatMessage(BaseModel):
    role: str
    content: str
    model: str | None = None
    created_at: str | None = None


class ConversationMessagesResponse(BaseModel):
    conversation_id: int
    messages: list[ChatMessage] = []


class ChatModel(BaseModel):
    id: str  # the litellm model id
    label: str
    provider: str


class ChatModelsResponse(BaseModel):
    models: list[ChatModel] = []


class KeyMeta(BaseModel):
    provider: str
    label: str | None = None
    created_at: str | None = None


class KeysResponse(BaseModel):
    keys: list[KeyMeta] = []


class StoreKeyRequest(BaseModel):
    provider: str
    api_key: str
    label: str | None = None


class MutationResponse(BaseModel):
    ok: bool
    message: str = ""


class SavedPrompt(BaseModel):
    id: int
    name: str
    text: str
    created_at: str | None = None


class SavedPromptsResponse(BaseModel):
    prompts: list[SavedPrompt] = []


class CreatePromptRequest(BaseModel):
    name: str
    text: str
