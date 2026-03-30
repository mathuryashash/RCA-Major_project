import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import yaml
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    username: str
    disabled: bool = False
    scopes: list[str] = []


class UserInDB(User):
    hashed_password: str


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config.yaml"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_jwt_settings():
    config = load_config()
    jwt_config = config.get("jwt", {})
    return {
        "secret_key": os.getenv(
            "JWT_SECRET_KEY", jwt_config.get("secret_key", SECRET_KEY)
        ),
        "algorithm": jwt_config.get("algorithm", ALGORITHM),
        "access_token_expire_minutes": jwt_config.get(
            "access_token_expire_minutes", ACCESS_TOKEN_EXPIRE_MINUTES
        ),
        "refresh_token_expire_days": jwt_config.get(
            "refresh_token_expire_days", REFRESH_TOKEN_EXPIRE_DAYS
        ),
    }


USERS_DB: dict[str, dict] = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def init_default_users():
    default_users = [
        {"username": "admin", "password": "admin123", "scopes": ["admin"]},
        {
            "username": "operator",
            "password": "operator123",
            "scopes": ["read", "write"],
        },
        {"username": "viewer", "password": "viewer123", "scopes": ["read"]},
    ]
    for user_data in default_users:
        username = user_data["username"]
        hashed = get_password_hash(user_data["password"])
        USERS_DB[username] = {
            "username": username,
            "hashed_password": hashed,
            "disabled": False,
            "scopes": user_data["scopes"],
        }


init_default_users()


def get_user(username: str) -> Optional[UserInDB]:
    if username in USERS_DB:
        user_dict = USERS_DB[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    settings = get_jwt_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings["access_token_expire_minutes"])
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(
        to_encode, settings["secret_key"], algorithm=settings["algorithm"]
    )


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    settings = get_jwt_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=settings["refresh_token_expire_days"])
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(
        to_encode, settings["secret_key"], algorithm=settings["algorithm"]
    )


def verify_token(token: str, token_type: str = "access") -> dict:
    settings = get_jwt_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings["secret_key"], algorithms=[settings["algorithm"]]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError:
        raise credentials_exception


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    payload = verify_token(token, "access")
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(username=user.username, disabled=user.disabled, scopes=user.scopes)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
