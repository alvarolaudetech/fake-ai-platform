"""Authentication and authorization logic.

Supports two credential types:
  * JWT bearer tokens (dashboard/session login via email+password)
  * API keys (server-to-server, used by the /inference endpoint)

Both resolve to a `User` ORM object via FastAPI dependencies so route
handlers never need to know which credential type was presented.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import get_settings
from core.database import APIKey, User, get_db

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_v1_prefix}/auth/token", auto_error=False)
api_key_scheme = APIKeyHeader(name=settings.api_key_header_name, auto_error=False)


class AuthError(HTTPException):
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    payload = {"sub": subject, "exp": expire, "type": "refresh"}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise AuthError("Token is invalid or expired") from exc


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.query(User).filter(User.email == email, User.is_active.is_(True)).first()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


# --- API key helpers -------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Return (raw_key, prefix, hashed_key) for a freshly minted API key.

    Only the hash is persisted; the raw key is shown to the user exactly once.
    """
    raw_key = f"fak_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:12]
    hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, prefix, hashed


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def resolve_api_key(db: Session, raw_key: str) -> User:
    hashed = _hash_api_key(raw_key)
    record = (
        db.query(APIKey)
        .filter(APIKey.hashed_key == hashed, APIKey.is_revoked.is_(False))
        .first()
    )
    if record is None:
        raise AuthError("Invalid API key")

    record.last_used_at = datetime.now(timezone.utc)
    db.commit()

    if not record.owner.is_active:
        raise AuthError("Owner account is disabled")
    return record.owner


# --- FastAPI dependencies ---------------------------------------------------

def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    api_key: str | None = Depends(api_key_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Accept either a JWT bearer token or an API key, in that order."""
    if api_key:
        return resolve_api_key(db, api_key)

    if token:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise AuthError("Expected an access token")
        user = db.query(User).filter(User.id == payload.get("sub")).first()
        if user is None or not user.is_active:
            raise AuthError("User not found or inactive")
        return user

    raise AuthError("Missing bearer token or API key")


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role.value != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user
