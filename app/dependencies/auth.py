from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
from typing import Optional


security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

class TokenData:
    def __init__(self, email: str, user_id: Optional[str] = None, name: Optional[str] = None):
        self.email = email
        self.user_id = user_id
        self.name = name

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    Verify JWT token and extract user information
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "success": False,
            "message": "Could not validate credentials",
            "error": "AUTHENTICATION_ERROR"
        },
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "JWT configuration error",
                "error": "SERVER_ERROR"
            }
        )

    try:
        token = credentials.credentials
        # Check if token looks like a JWT (should have 3 parts separated by dots)
        token_parts = token.split('.')
        if len(token_parts) != 3:
            raise credentials_exception
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("email")
        user_id: Optional[str] = payload.get("id") or payload.get("userId")
        name: Optional[str] = payload.get("name")

        if email is None:
            raise credentials_exception
        return TokenData(email=email, user_id=user_id, name=name)
    except JWTError:
        raise credentials_exception
    except Exception:
        raise credentials_exception