import datetime
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pymongo.database import Database
from config import settings
from database import get_db
import models

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# passlib (previously used here) is unmaintained since 2020 and its bcrypt
# backend self-test raises ValueError against bcrypt>=4.1 — it breaks login
# entirely on any recent install. Calling the bcrypt library directly avoids
# that dependency altogether.
#
# bcrypt has a hard 72-BYTE input limit (not 72 characters — multi-byte UTF-8
# passwords can exceed it well before 72 chars) and silently truncates or
# errors past that depending on version, so we truncate deliberately and
# consistently here rather than relying on the library's own behavior.
_BCRYPT_MAX_BYTES = 72


def _prepare(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(_prepare(password), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(plain), hashed.encode("utf-8"))
    except ValueError:
        # Malformed/foreign hash format in the DB — treat as failed auth,
        # not a 500.
        return False


def create_access_token(subject: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("purpose") is not None:
            return None  # a reset token was presented where a login token belongs
        return payload.get("sub")
    except JWTError:
        return None


RESET_TOKEN_EXPIRE_MINUTES = 30


def create_reset_token(subject: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire, "purpose": "password_reset"}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_reset_token(token: str) -> Optional[str]:
    """Returns the user email if `token` is a valid, unexpired reset token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("purpose") != "password_reset":
            return None
        return payload.get("sub")
    except JWTError:
        return None


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Database = Depends(get_db)) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_exception
    email = _decode_token(token)
    if email is None:
        raise credentials_exception
    doc = db.users.find_one({"email": email})
    if doc is None:
        raise credentials_exception
    return models.User(doc)


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme), db: Database = Depends(get_db)) -> Optional[models.User]:
    if token is None:
        return None
    email = _decode_token(token)
    if email is None:
        return None
    doc = db.users.find_one({"email": email})
    return models.User(doc) if doc else None