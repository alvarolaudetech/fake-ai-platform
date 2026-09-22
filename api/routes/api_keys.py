"""Endpoints for users to manage their own API keys."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.services.auth_service import generate_api_key, get_current_user
from core.database import APIKey, User, get_db

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class APIKeyOut(BaseModel):
    id: str
    key_prefix: str
    is_revoked: bool
    created_at: str

    class Config:
        from_attributes = True


class APIKeyCreateResponse(BaseModel):
    id: str
    raw_key: str
    key_prefix: str
    created_at: str


@router.post("", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a new API key. The full raw key is returned only once."""
    raw_key, prefix, hashed = generate_api_key()
    
    key_record = APIKey(
        key_prefix=prefix,
        hashed_key=hashed,
        label="Default Key",
        owner_id=current_user.id,
    )
    
    db.add(key_record)
    db.commit()
    db.refresh(key_record)
    
    return APIKeyCreateResponse(
        id=key_record.id,
        raw_key=raw_key,
        key_prefix=prefix,
        created_at=key_record.created_at.isoformat(),
    )


@router.get("", response_model=list[APIKeyOut])
def list_api_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all API keys belonging to the current user."""
    return db.query(APIKey).filter(APIKey.owner_id == current_user.id).all()


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke an API key by ID."""
    key_record = (
        db.query(APIKey)
        .filter(APIKey.id == key_id, APIKey.owner_id == current_user.id)
        .first()
    )
    
    if key_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="API key not found or you don't have permission to revoke it"
        )
    
    key_record.is_revoked = True
    db.commit()
    return None
