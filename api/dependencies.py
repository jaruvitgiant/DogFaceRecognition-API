"""
api/dependencies.py
--------------------
Dependency Injection container สำหรับ FastAPI.

หลักการ SOLID ที่ใช้:
  - DIP: Router พึ่ง interface ผ่าน Depends() ไม่ใช่ concrete class โดยตรง
  - SRP: ไฟล์นี้รับผิดชอบแค่การ wire service instances ให้ router

Pattern:
  - ทุก service เป็น singleton (สร้างครั้งเดียวตอน startup)
  - Router เรียกใช้ผ่าน get_*() functions ที่เป็น FastAPI Depends
"""

from functools import lru_cache

from infrastructure.model_registry import ModelRegistry
from infrastructure.image_processor import YoloCropProcessor
from services.embedding_service import EmbeddingService
from services.search_service import KNNSearchService
from services.training_service import TrainingOrchestrator
from services.visualization_service import VisualizationService


# ── Singleton instances ───────────────────────────────────────────────
# สร้างครั้งเดียวที่ module load time

_model_registry = ModelRegistry()
_image_processor = YoloCropProcessor()
_embedding_service = EmbeddingService(_model_registry, _image_processor)
_search_service = KNNSearchService(_model_registry, _image_processor)
_training_orchestrator = TrainingOrchestrator(_image_processor)
_visualization_service = VisualizationService()


# ── FastAPI Dependency getters ────────────────────────────────────────
# ใช้ใน router: service: ModelRegistry = Depends(get_model_registry)

def get_model_registry() -> ModelRegistry:
    return _model_registry


def get_image_processor() -> YoloCropProcessor:
    return _image_processor


def get_embedding_service() -> EmbeddingService:
    return _embedding_service


def get_search_service() -> KNNSearchService:
    return _search_service


def get_training_orchestrator() -> TrainingOrchestrator:
    return _training_orchestrator


def get_visualization_service() -> VisualizationService:
    return _visualization_service
