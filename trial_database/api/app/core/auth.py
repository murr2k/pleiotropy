"""
Authentication utilities
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import settings
from app.db.database import get_db
from app.models.database import Agent
from app.models.schemas import TokenData

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/agents/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_agent(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> Agent:
    """Get the current authenticated agent"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        agent_id: int = payload.get("sub")
        if agent_id is None:
            raise credentials_exception
        token_data = TokenData(agent_id=agent_id)
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(
        select(Agent).where(Agent.id == token_data.agent_id)
    )
    agent = result.scalar_one_or_none()
    
    if agent is None:
        raise credentials_exception
    
    if not agent.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent account is deactivated"
        )
    
    # Update last active timestamp
    agent.last_active = datetime.utcnow()
    await db.commit()
    
    return agent


async def get_current_active_agent(
    current_agent: Agent = Depends(get_current_agent)
) -> Agent:
    """Ensure the agent is active"""
    if not current_agent.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive agent"
        )
    return current_agent


def require_role(required_role: str):
    """Dependency to require a specific agent role"""
    async def role_checker(current_agent: Agent = Depends(get_current_active_agent)):
        if current_agent.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Agent role '{required_role}' required"
            )
        return current_agent
    return role_checker