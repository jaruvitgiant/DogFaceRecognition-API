"""
app.py — Application Entry Point (ใหม่)
-----------------------------------------
สร้าง FastAPI application โดยรวม routers ทั้งหมดเข้าด้วยกัน
และตั้งค่า lifespan, middleware

หลักการ SOLID:
  - SRP: ไฟล์นี้รับผิดชอบแค่ App assembly/wiring
  - OCP: เพิ่ม router ใหม่ได้โดยไม่แก้ core logic
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from api.dependencies import get_model_registry
from api.routers import (
    model_router,
    embedding_router,
    search_router,
    training_router,
    knn_router,
)


# ── Lifespan: startup / shutdown ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: โหลด default model weight เพื่อให้พร้อมใช้งานทันที
    Shutdown: log และ cleanup (ถ้ามี)
    """
    registry = get_model_registry()
    print(f"🚀 Startup: Loading default model from {settings.DEFAULT_MODEL_PATH}")

    success = registry.load(settings.DEFAULT_MODEL_PATH)
    if success:
        print(f"✅ Default model loaded: {registry.get_active_name()}")
    else:
        print("⚠️  Default model not found — API still running, load model via /model/select-model")

    yield

    print("🛑 Shutdown complete.")


# ── App Factory ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Factory function สร้าง FastAPI instance
    (ทดสอบได้ง่ายกว่าการสร้าง global app โดยตรง)
    """
    application = FastAPI(
        title="Dog Face Recognition API",
        description="FastAPI service สำหรับ dog face embedding, search และ training",
        version="2.0.0",
        lifespan=lifespan,
    )

    # ── CORS Middleware ────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register Routers ──────────────────────────────────────────
    application.include_router(model_router.router)
    application.include_router(embedding_router.router)
    application.include_router(search_router.router)
    application.include_router(training_router.router)
    application.include_router(knn_router.router)

    return application


# ── Singleton App Instance ───────────────────────────────────────────
app = create_app()
