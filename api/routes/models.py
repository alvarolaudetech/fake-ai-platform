"""REST endpoints for managing the catalog of registered LLM backends.

Regular users can list/read active models to know what they can call from
/inference. Only admins can register, update, or deactivate a model.
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.services.auth_service import get_current_user, require_admin
from core.database import ModelConfig, ModelProvider, User, get_db

router = APIRouter(prefix="/models", tags=["models"])


class ModelConfigCreate(BaseModel):
    model_slug: str = Field(..., examples=["gpt-4o-mini"])
    display_name: str
    provider: ModelProvider
    upstream_model_name: str
    context_window: int = 8192
    max_output_tokens: int = 1024
    input_cost_per_1M: float = 0.0
    output_cost_per_1M: float = 0.0
    supports_streaming: bool = True
    system_prompt_override: str | None = None


class ModelConfigUpdate(BaseModel):
    display_name: str | None = None
    max_output_tokens: int | None = None
    input_cost_per_1M: float | None = None
    output_cost_per_1M: float | None = None
    is_active: bool | None = None
    supports_streaming: bool | None = None
    system_prompt_override: str | None = None


class ModelConfigOut(BaseModel):
    id: str
    model_slug: str
    display_name: str
    provider: ModelProvider
    upstream_model_name: str
    context_window: int
    max_output_tokens: int
    input_cost_per_1M: float
    output_cost_per_1M: float
    is_active: bool
    supports_streaming: bool

    class Config:
        from_attributes = True


@router.get("", response_model=list[ModelConfigOut])
def list_models(
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(ModelConfig)
    if not include_inactive:
        query = query.filter(ModelConfig.is_active.is_(True))
    return query.order_by(ModelConfig.display_name).all()


@router.get("/{model_slug}", response_model=ModelConfigOut)
def get_model(
    model_slug: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model = db.query(ModelConfig).filter(ModelConfig.model_slug == model_slug).first()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.post("", response_model=ModelConfigOut, status_code=status.HTTP_201_CREATED)
def create_model(
    payload: ModelConfigCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    existing = db.query(ModelConfig).filter(ModelConfig.model_slug == payload.model_slug).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"model_slug '{payload.model_slug}' already registered",
        )

    model = ModelConfig(**payload.model_dump())
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.patch("/{model_slug}", response_model=ModelConfigOut)
def update_model(
    model_slug: str,
    payload: ModelConfigUpdate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    model = db.query(ModelConfig).filter(ModelConfig.model_slug == model_slug).first()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(model, field, value)

    db.commit()
    db.refresh(model)
    return model


@router.delete("/{model_slug}", status_code=status.HTTP_204_NO_CONTENT)
def deactivate_model(
    model_slug: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Soft-delete: models are deactivated rather than removed to preserve
    referential integrity with historical inference_logs rows."""
    model = db.query(ModelConfig).filter(ModelConfig.model_slug == model_slug).first()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    model.is_active = False
    db.commit()
