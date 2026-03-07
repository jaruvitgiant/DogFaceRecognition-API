"""
api/routers/model_router.py
-----------------------------
Router สำหรับจัดการ model versions.
Endpoints: GET /models, GET /current-model, POST /select-model

หลักการ SOLID:
  - SRP: Router รับผิดชอบแค่ HTTP layer — delegate ทุกอย่างไปหา ModelRegistry
  - DIP: พึ่งพา ModelRegistry ผ่าน Depends() ไม่ hard-code
"""

from fastapi import APIRouter, HTTPException, Depends

from infrastructure.model_registry import ModelRegistry
from api.dependencies import get_model_registry
from core.config import settings

router = APIRouter(prefix="/model", tags=["Model Management"])


@router.get("/models", summary="List ทุก model version ที่มีอยู่")
def list_models(registry: ModelRegistry = Depends(get_model_registry)):
    return {"models": registry.list_checkpoints()}


@router.get("/current-model", summary="ดู model ที่ active อยู่ตอนนี้")
def current_model(registry: ModelRegistry = Depends(get_model_registry)):
    return {
        "active_model": registry.get_active_name(),
        "active_path": registry.get_active_path(),
        "device": settings.DEVICE,
    }


@router.post("/select-model", summary="เลือก/สลับ model version")
def select_model(
    version: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    model_path = registry.resolve_path(version)

    if not __import__("os").path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบไฟล์โมเดล: {version} (ตรวจสอบที่: {model_path})",
        )

    success = registry.load(model_path)
    if not success:
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการโหลด Weights")

    return {
        "status": "success",
        "active_model": version,
        "path": model_path,
    }
