"""Bubba chat contracts (Phase B1). Request/response models for /api/chat/*."""

from __future__ import annotations

from pydantic import BaseModel


class ChatSendRequest(BaseModel):
    message: str
    model: str
    conversation_id: int | None = None
    web_search: bool = False
    deep_research: bool = False


class ChatSendResponse(BaseModel):
    content: str
    conversation_id: int | None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_trace: list[dict] = []
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
