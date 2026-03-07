"""
infrastructure/model_registry.py
---------------------------------
Concrete implementation ของ IModelRegistry.

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่ model lifecycle (load, switch, list)
  - OCP: เพิ่ม model type ใหม่ได้โดย extend class ไม่ต้องแก้ core
  - DIP: Implements IModelRegistry ทำให้ router พึ่ง interface แทน class นี้โดยตรง
"""

import os
import json
import torch
from typing import Optional

from core.interfaces import IModelRegistry
from core.config import settings
from resnet.resnet import ResNetBackbone, Bottleneck


class ModelRegistry(IModelRegistry):
    """
    จัดการ lifecycle ของ ResNet model:
    - โหลด / สลับ weights
    - เก็บ state (active model, path, name) แบบ encapsulated
    - list versions ที่มีอยู่ใน checkpoint directory
    """

    def __init__(self) -> None:
        self._model: Optional[ResNetBackbone] = None
        self._active_path: Optional[str] = None
        self._active_name: Optional[str] = None
        self._device: str = settings.DEVICE

        # สร้าง model architecture เพียงครั้งเดียว (ไม่สร้างซ้ำทุกครั้งที่โหลด weight)
        self._model = ResNetBackbone(
            Bottleneck,
            [3, 8, 36, 3],
            embedding_size=settings.EMBEDDING_SIZE
        )

    # ───────────────────────────────────────────────
    # IModelRegistry contract
    # ───────────────────────────────────────────────

    def load(self, model_path: str) -> bool:
        """โหลด state_dict จาก path เข้า model ที่สร้างไว้แล้ว"""
        if not os.path.exists(model_path):
            print(f" ModelRegistry: model path not found → {model_path}")
            return False

        try:
            state_dict = torch.load(model_path, map_location=self._device)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            self._active_path = model_path
            self._active_name = os.path.basename(os.path.dirname(model_path))
            print(f" ModelRegistry: loaded → {model_path}")
            return True

        except Exception as exc:
            print(f" ModelRegistry: failed to load model — {exc}")
            return False

    def get_model(self) -> Optional[ResNetBackbone]:
        """คืน model instance (ใช้ใน service layer)"""
        return self._model

    def get_active_name(self) -> Optional[str]:
        return self._active_name

    def get_active_path(self) -> Optional[str]:
        return self._active_path

    # ───────────────────────────────────────────────
    # Extra helper (ใช้โดย model_router)
    # ───────────────────────────────────────────────

    def list_checkpoints(self) -> list[dict]:
        """
        List ทุก checkpoint ที่มีใน BASE_CHECKPOINT_DIR
        คืน list ของ dict ที่ประกอบด้วย id, path, active, details
        """
        models = []

        # Default model
        models.append({
            "id": "default",
            "type": "legacy",
            "path": settings.DEFAULT_MODEL_PATH,
            "active": self._active_path == settings.DEFAULT_MODEL_PATH,
        })

        # Versioned models
        base_dir = settings.BASE_CHECKPOINT_DIR
        if os.path.exists(base_dir):
            for version_dir in sorted(os.listdir(base_dir)):
                version_path = os.path.join(base_dir, version_dir)
                model_path = os.path.join(version_path, "model.pth")
                meta_path = os.path.join(version_path, "meta.json")

                if not os.path.exists(model_path):
                    continue

                info: dict = {
                    "id": version_dir,
                    "type": "versioned",
                    "path": model_path,
                    "active": self._active_path == model_path,
                }

                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            info["details"] = json.load(f)
                    except json.JSONDecodeError:
                        info["details"] = {"error": "invalid meta.json"}

                models.append(info)

        return models

    def resolve_path(self, version: str) -> str:
        """
        แปลง version string เป็น model path จริง
        'default' → DEFAULT_MODEL_PATH
        อื่นๆ → BASE_CHECKPOINT_DIR/<version>/model.pth
        """
        if version == "default":
            return settings.DEFAULT_MODEL_PATH
        return os.path.join(settings.BASE_CHECKPOINT_DIR, version, "model.pth")
