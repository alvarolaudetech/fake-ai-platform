from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from core.config import get_settings
from core.database import InferenceLog, UsageQuota, User

settings = get_settings()

class QuotaService:
    def __init__(self, db: Session):
        self.db = db

    def enforce_quota(self, user: User, estimated_tokens: int) -> UsageQuota:
        quota = self.db.query(UsageQuota).filter(UsageQuota.user_id == user.id).first()
        if quota is None:
            quota = UsageQuota(user_id=user.id, daily_token_limit=settings.default_daily_token_quota)
            self.db.add(quota)
            self.db.commit()
            self.db.refresh(quota)

        if datetime.now(timezone.utc) - quota.reset_at > timedelta(days=1):
            quota.tokens_used_today = 0
            quota.reset_at = datetime.now(timezone.utc)
            self.db.commit()

        if quota.tokens_used_today + estimated_tokens > quota.daily_token_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily token quota exceeded",
            )
        return quota

    def record_usage(self, quota: UsageQuota, tokens_used: int) -> None:
        quota.tokens_used_today += tokens_used
        self.db.commit()

    def log_inference(
        self,
        user: User,
        model_slug: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        status_code: int,
        error_message: str | None = None,
    ) -> None:
        self.db.add(
            InferenceLog(
                user_id=user.id,
                model_slug=model_slug,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status_code=status_code,
                error_message=error_message,
            )
        )
        self.db.commit()
