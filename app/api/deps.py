"""Shared API dependencies: auth, rate limiting."""
import hashlib
import hmac
import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)
limiter = Limiter(key_func=get_remote_address)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Verify bearer token against API_KEY_HASH. Skip if hash not configured."""
    s = get_settings()
    if not s.API_KEY_HASH:
        return None
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    if not hmac.compare_digest(token_hash, s.API_KEY_HASH):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
