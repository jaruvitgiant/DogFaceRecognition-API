"""
core/config.py
--------------
Centralized configuration management โดยใช้ Pydantic Settings.

หลักการ SOLID ที่ใช้:
  - SRP: Config มีหน้าที่เดียวคือจัดเก็บ/โหลด configuration
  - DIP: Component อื่นๆ พึ่งพา Settings object ไม่ใช่ hard-coded strings
"""

import torch
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Root directory ของ project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Application settings — โหลดจาก environment variables / .env file
    ใช้ Pydantic BaseSettings เพื่อ validate ค่าและ support .env
    """

    # ─── Model Paths ───────────────────────────────────────────
    DEFAULT_MODEL_PATH: str = Field(
        default="/home/jaruvitgitant/Documents/Project_findmydog/api-modelfaceNetDog/fastapi-models/resnet/model_resnet152_default/resne152-V01_60.pt",
        description="Path ของ model เริ่มต้น"
    )
    BASE_CHECKPOINT_DIR: str = Field(
        default=str(BASE_DIR / "checkpoints" / "resnet152"),
        description="Directory สำหรับเก็บ checkpoint weights"
    )

    # ─── KNN Paths ──────────────────────────────────────────────
    KNN_MODEL_PATH: str = Field(
        default="models/knn_latest02.joblib",
        description="Path สำหรับบันทึก/โหลด KNN model"
    )
    KNN_LABELS_PATH: str = Field(
        default="models/labels_latest02.joblib",
        description="Path สำหรับบันทึก/โหลด KNN labels"
    )

    # ─── YOLO ───────────────────────────────────────────────────
    YOLO_MODEL_PATH: str = Field(
        default="/home/jaruvitgitant/Documents/Project_findmydog/AI/Detect_dog_faceYOLO11/runs/detect/train3/weights/best.pt",
        description="Path ของ YOLO model สำหรับ dog face detection"
    )
    YOLO_TARGET_CLASS: str = Field(
        default="FaceDog",
        description="ชื่อ class ที่ YOLO จะ detect"
    )
    YOLO_CONF_THRESHOLD: float = Field(
        default=0.6,
        description="Confidence threshold สำหรับ YOLO detection"
    )

    # ─── ResNet ─────────────────────────────────────────────────
    EMBEDDING_SIZE: int = Field(
        default=512,
        description="ขนาดของ embedding vector"
    )
    SEARCH_HISTORY_DIR: str = Field(
        default="search_history",
        description="Directory สำหรับเก็บภาพที่ค้นหา"
    )

    # ─── Auth ───────────────────────────────────────────────────
    AUTO_TRAIN_SECRET: str = Field(
        default="",
        description="JWT secret key สำหรับ /retrain-model-face"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    # ─── CORS ───────────────────────────────────────────────────
    ALLOWED_ORIGINS: list = Field(
        default=["http://127.0.0.1:8000", "http://localhost:8000"],
        description="Allowed origins สำหรับ CORS"
    )

    # ─── Runtime (computed) ─────────────────────────────────────
    @property
    def DEVICE(self) -> str:
        """คืน 'cuda' ถ้ามี GPU ไม่งั้นคืน 'cpu'"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance — ใช้ทั่วทั้ง application
settings = Settings()
