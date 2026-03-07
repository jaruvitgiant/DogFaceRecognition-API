# app/auth.py
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

load_dotenv()  # โหลด .env ครั้งเดียวที่นี่

AUTO_TRAIN_SECRET = os.getenv("AUTO_TRAIN_SECRET")

if not AUTO_TRAIN_SECRET:
    raise RuntimeError("AUTO_TRAIN_SECRET is not set in environment")
security = HTTPBearer()
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, AUTO_TRAIN_SECRET, algorithms=[ALGORITHM])

        if payload.get("scope") != "auto_retrain":
            raise HTTPException(status_code=403, detail="Invalid scope")

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
