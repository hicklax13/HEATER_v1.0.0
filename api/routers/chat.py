"""Bubba chat router (Phase B1). Thin + logic-free: each route resolves the
Clerk AppUser, derives the namespaced chat user_id, and delegates to ChatService.
Chat is inherently per-user, so an unauthenticated caller (no Clerk) gets 401."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.contracts.chat import (
    ChatModelsResponse,
    ChatSendRequest,
    ChatSendResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    KeysResponse,
    MutationResponse,
    StoreKeyRequest,
)
from api.deps import get_chat_service
from api.identity import require_app_user
from api.services.chat_service import ChatService, chat_user_id_for
from api.stores.user_store import AppUser

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _uid(app_user: AppUser | None) -> int:
    if app_user is None:
        raise HTTPException(status_code=401, detail="Sign in to use Bubba.")
    return chat_user_id_for(app_user.id)


@router.post("/send", response_model=ChatSendResponse)
def send(
    body: ChatSendRequest,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> ChatSendResponse:
    return ChatSendResponse(
        **svc.send(
            chat_user_id=_uid(app_user),
            message=body.message,
            model=body.model,
            conversation_id=body.conversation_id,
            web_search=body.web_search,
            deep_research=body.deep_research,
            reasoning_effort=body.reasoning_effort,
            attached_text=body.attached_text,
            attachments=body.attachments,
        )
    )


@router.post("/send-stream")
def send_stream(
    body: ChatSendRequest,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        svc.send_stream(
            chat_user_id=_uid(app_user),
            message=body.message,
            model=body.model,
            conversation_id=body.conversation_id,
            web_search=body.web_search,
            deep_research=body.deep_research,
            reasoning_effort=body.reasoning_effort,
            attached_text=body.attached_text,
            attachments=body.attachments,
        ),
        media_type="text/event-stream",
    )


@router.get("/conversations", response_model=ConversationListResponse)
def conversations(
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> ConversationListResponse:
    return ConversationListResponse(conversations=svc.conversations(_uid(app_user)))


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def messages(
    conversation_id: int,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> ConversationMessagesResponse:
    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=svc.messages(_uid(app_user), conversation_id),
    )


@router.get("/models", response_model=ChatModelsResponse)
def models(
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> ChatModelsResponse:
    return ChatModelsResponse(models=svc.models(_uid(app_user)))


@router.get("/keys", response_model=KeysResponse)
def list_keys(
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> KeysResponse:
    return KeysResponse(keys=svc.list_keys(_uid(app_user)))


@router.put("/keys", response_model=MutationResponse)
def put_key(
    body: StoreKeyRequest,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> MutationResponse:
    ok, msg = svc.store_key(_uid(app_user), body.provider, body.api_key, body.label)
    return MutationResponse(ok=ok, message=msg)


@router.delete("/keys", response_model=MutationResponse)
def delete_key(
    provider: str,
    label: str | None = None,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> MutationResponse:
    ok, msg = svc.delete_key(_uid(app_user), provider, label)
    return MutationResponse(ok=ok, message=msg)
