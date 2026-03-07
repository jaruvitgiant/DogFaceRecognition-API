"""
services/embedding_service.py
------------------------------
Business logic สำหรับแปลงภาพสุนัขเป็น embedding vector.

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่ embedding — ไม่ยุ่งกับ search/training
  - DIP: พึ่งพา IModelRegistry และ IImageProcessor (interfaces)
         ไม่ได้ coupled กับ ResNet หรือ YOLO โดยตรง
"""

import io
import base64
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image

from core.interfaces import IModelRegistry, IImageProcessor, IEmbeddingModel
from core.config import settings


# ── Image transform pipeline (ImageNet normalization) ────────────────
def build_transform() -> A.Compose:
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class EmbeddingService(IEmbeddingModel):
    """
    แปลงภาพสุนัขเป็น embedding vector โดยใช้ ResNet backbone
    พึ่งพา IModelRegistry เพื่อ access model ที่ active อยู่
    พึ่งพา IImageProcessor เพื่อ crop ใบหน้าก่อนส่งเข้า model
    """

    def __init__(
        self,
        model_registry: IModelRegistry,
        image_processor: IImageProcessor,
    ) -> None:
        self._registry = model_registry
        self._processor = image_processor
        self._transform = build_transform()
        self._device = settings.DEVICE

    # ───────────────────────────────────────────────
    # IEmbeddingModel contract
    # ───────────────────────────────────────────────

    def get_embedding(self, img_pil: Image.Image) -> np.ndarray:
        """
        แปลง PIL Image เป็น embedding vector (numpy 1D array)
        """
        model = self._registry.get_model()
        if model is None:
            raise RuntimeError("No model loaded in ModelRegistry")

        model.eval()
        image_np = np.array(img_pil)
        augmented = self._transform(image=image_np)
        image_tensor = augmented["image"].unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy().flatten()

        return embedding

    # ───────────────────────────────────────────────
    # Higher-level helpers (ใช้โดย router)
    # ───────────────────────────────────────────────

    async def process_upload_file(self, file) -> dict:
        """
        รับ FastAPI UploadFile
        คืน dict ที่ประกอบด้วย filename, embedding_dim, embedding_base64
        """
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # YOLO crop (ใช้ภาพเดิมถ้า crop ไม่ได้)
        cropped = self._processor.process(img_pil) or img_pil
        emb = self.get_embedding(cropped)

        emb_bytes = emb.astype(np.float32).tobytes()
        emb_base64 = base64.b64encode(emb_bytes).decode("utf-8")

        return {
            "filename": file.filename,
            "embedding_dim": len(emb),
            "embedding_base64": emb_base64,
        }

    def get_tensor_from_image(self, img_pil: Image.Image):
        """
        คืน tensor ที่พร้อมใช้กับ model (ใช้ใน search service)
        """
        image_np = np.array(img_pil)
        augmented = self._transform(image=image_np)
        return augmented["image"].unsqueeze(0).to(self._device)
