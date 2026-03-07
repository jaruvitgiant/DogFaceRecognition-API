"""
infrastructure/auth_service.py
--------------------------------
Concrete implementation ของ IAuthService โดยใช้ JWT.

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่ JWT verification ไม่ยุ่งกับ business logic
  - DIP: Implements IAuthService — ถ้าอยากเปลี่ยนเป็น OAuth2 ให้ extend ใหม่
"""

import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.interfaces import IAuthService
from core.config import settings

_security = HTTPBearer()


class JWTAuthService(IAuthService):
    """
    ตรวจสอบ JWT Bearer token และ scope ที่กำหนด
    """

    def __init__(
        self,
        secret: str = settings.AUTO_TRAIN_SECRET,
        algorithm: str = settings.JWT_ALGORITHM,
        required_scope: str = "auto_retrain",
    ) -> None:
        self._secret = secret
        self._algorithm = algorithm
        self._required_scope = required_scope

    # ───────────────────────────────────────────────
    # IAuthService contract
    # ───────────────────────────────────────────────

    def verify(self, token: str) -> dict:
        """
        Decode และ validate JWT token
        raise HTTPException ถ้า token ไม่ valid หรือ scope ไม่ตรง
        """
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])

            if payload.get("scope") != self._required_scope:
                raise HTTPException(status_code=403, detail="Invalid scope")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")


# ── FastAPI Dependency helper ────────────────────────────────────
# ใช้แบบนี้ใน router: payload=Depends(verify_token_dep)

_auth_service = JWTAuthService()


def verify_token_dep(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """FastAPI dependency function — ใช้แทน verify_token เดิม"""
    return _auth_service.verify(credentials.credentials)
