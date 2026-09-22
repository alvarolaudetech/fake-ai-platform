"""The core inference endpoint: chat completions routed through the
registered model catalog, with quota enforcement and usage logging.
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.services.auth_service import get_current_user
from api.services.llm_service import ChatMessage, LLMServiceError, get_llm_service
from api.services.quota_service import QuotaService
from core.config import get_settings
from core.database import ModelConfig, User, get_db

router = APIRouter(prefix="/inference", tags=["inference"])
settings = get_settings()


class ChatMessageIn(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="model_slug of a registered ModelConfig")
    messages: list[ChatMessageIn]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    finish_reason: str
    estimated_cost_usd: float


def _get_active_model(db: Session, model_slug: str) -> ModelConfig:
    model = (
        db.query(ModelConfig)
        .filter(ModelConfig.model_slug == model_slug, ModelConfig.is_active.is_(True))
        .first()
    )
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active model registered under slug '{model_slug}'",
        )
    return model


# (Nothing here, these functions were removed)


@router.post("/chat", response_model=ChatCompletionResponse)
def chat_completion(
    payload: ChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model = _get_active_model(db, payload.model)
    llm_service = get_llm_service()
    quota_service = QuotaService(db)
    messages = [ChatMessage(role=m.role, content=m.content) for m in payload.messages]

    estimated_prompt_tokens = sum(llm_service.count_tokens(m.content) for m in messages)
    quota = quota_service.enforce_quota(current_user, estimated_prompt_tokens + (payload.max_tokens or model.max_output_tokens))

    try:
        result = llm_service.complete(
            model=model,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except LLMServiceError as exc:
        quota_service.log_inference(current_user, model.model_slug, estimated_prompt_tokens, 0, 0, 502, str(exc))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    total_tokens = result.prompt_tokens + result.completion_tokens
    quota_service.record_usage(quota, total_tokens)
    quota_service.log_inference(
        current_user,
        model.model_slug,
        result.prompt_tokens,
        result.completion_tokens,
        result.latency_ms,
        200,
    )

    return ChatCompletionResponse(
        model=model.model_slug,
        content=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=total_tokens,
        latency_ms=result.latency_ms,
        finish_reason=result.finish_reason,
        estimated_cost_usd=llm_service.estimate_cost_usd(model, result.prompt_tokens, result.completion_tokens),
    )


@router.post("/chat/stream")
async def chat_completion_stream(
    payload: ChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model = _get_active_model(db, payload.model)
    if not model.supports_streaming:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model.model_slug}' does not support streaming responses",
        )

    llm_service = get_llm_service()
    quota_service = QuotaService(db)
    messages = [ChatMessage(role=m.role, content=m.content) for m in payload.messages]

    estimated_prompt_tokens = sum(llm_service.count_tokens(m.content) for m in messages)
    quota = quota_service.enforce_quota(current_user, estimated_prompt_tokens + (payload.max_tokens or model.max_output_tokens))

    async def event_generator():
        collected_chunks: list[str] = []
        try:
            async for chunk in llm_service.stream(
                model=model,
                messages=messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            ):
                collected_chunks.append(chunk)
                yield f"data: {chunk}\n\n"
        except LLMServiceError as exc:
            yield f"event: error\ndata: {exc}\n\n"
            return
        finally:
            # Use a new DB session for final logging as the request session may be closed/invalidated during streaming
            from core.database import get_db, UsageQuota
            with next(get_db()) as db_session:
                quota_service_final = QuotaService(db_session)
                completion_tokens = llm_service.count_tokens("".join(collected_chunks))
                user_quota = db_session.query(UsageQuota).filter(UsageQuota.user_id == current_user.id).first()
                if user_quota:
                    quota_service_final.record_usage(user_quota, estimated_prompt_tokens + completion_tokens)
                quota_service_final.log_inference(
                    current_user, model.model_slug, estimated_prompt_tokens, completion_tokens, 0, 200
                )
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
