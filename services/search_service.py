"""
services/search_service.py
---------------------------
Business logic สำหรับ KNN-based dog search.

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่ KNN index management และ search
  - OCP: เปลี่ยน algorithm (เช่น FAISS) ได้โดย subclass ใหม่
  - LSP: Implements ISearchService ทำให้ใช้แทน implementation อื่นได้
  - DIP: Router พึ่งพา ISearchService ไม่ใช่ KNNSearchService โดยตรง
"""

import os
import io
import uuid
import base64
import numpy as np
import joblib
import torch
from datetime import datetime
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional

from core.interfaces import ISearchService, IModelRegistry, IImageProcessor
from core.config import settings


class KNNSearchService(ISearchService):
    """
    ใช้ sklearn NearestNeighbors (cosine metric) สำหรับค้นหาสุนัข
    """

    def __init__(
        self,
        model_registry: IModelRegistry,
        image_processor: IImageProcessor,
        knn_model_path: str = settings.KNN_MODEL_PATH,
        knn_labels_path: str = settings.KNN_LABELS_PATH,
    ) -> None:
        self._registry = model_registry
        self._processor = image_processor
        self._knn_model_path = knn_model_path
        self._knn_labels_path = knn_labels_path
        self._knn: Optional[NearestNeighbors] = None
        self._labels: Optional[list] = None
        self._device = settings.DEVICE

        # Transform pipeline (ImageNet)
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self._transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    # ───────────────────────────────────────────────
    # ISearchService contract
    # ───────────────────────────────────────────────

    def train(self, embeddings: List[np.ndarray], labels: List) -> None:
        """เทรน KNN index ใหม่"""
        X = np.vstack(embeddings)
        self._knn = NearestNeighbors(n_neighbors=len(X), metric="cosine")
        self._knn.fit(X)
        self._labels = labels
        
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        ค้นหา top_k สุนัขที่ใกล้เคียงที่สุด
        คืน list ของ {"rank": ..., "dog_id": ..., "distance": ...}
        """
        if not self.is_ready():
            raise RuntimeError("KNN model not trained. Call /tiger_knnTrain/ first.")

        query = embedding.reshape(1, -1)
        distances, indices = self._knn.kneighbors(query)

        # Deduplicate: เก็บ distance ที่ดีที่สุดต่อ dog
        unique_results: dict = {}
        for i, idx in enumerate(indices[0]):
            dog_id = self._labels[idx]
            dist = float(distances[0][i])
            if dog_id not in unique_results or dist < unique_results[dog_id]:
                unique_results[dog_id] = dist

        # Sort และ limit
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1])[:top_k]

        return [
            {"rank": rank + 1, "dog_id": dog_id, "distance": dist}
            for rank, (dog_id, dist) in enumerate(sorted_results)
        ]

    def save(self) -> None:
        """บันทึก KNN model และ labels ลง disk"""
        os.makedirs(os.path.dirname(self._knn_model_path) or "models", exist_ok=True)
        joblib.dump(self._knn, self._knn_model_path)
        joblib.dump(self._labels, self._knn_labels_path)

    def load(self) -> bool:
        """โหลด KNN model จาก disk"""
        if not os.path.exists(self._knn_model_path):
            return False
        try:
            self._knn = joblib.load(self._knn_model_path)
            self._labels = joblib.load(self._knn_labels_path)
            return True
        except Exception as exc:
            print(f"❌ KNNSearchService: failed to load — {exc}")
            return False

    def is_ready(self) -> bool:
        """ตรวจสอบว่า KNN model พร้อมใช้งาน"""
        return self._knn is not None and self._labels is not None

    # ───────────────────────────────────────────────
    # Helper: extract embedding จาก UploadFile
    # ───────────────────────────────────────────────

    async def extract_embedding_from_upload(self, file) -> Optional[np.ndarray]:
        """
        อ่านไฟล์อัปโหลด → YOLO crop → tensor → embedding
        คืน None ถ้า YOLO ตรวจไม่เจอสุนัข
        """
        import io as _io
        contents = await file.read()
        img_pil = Image.open(_io.BytesIO(contents)).convert("RGB")

        cropped = self._processor.process(img_pil)
        if cropped is None:
            return None

        # บันทึกภาพ crop ที่ค้นหา
        save_dir = settings.SEARCH_HISTORY_DIR
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"crop_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
        cropped.save(os.path.join(save_dir, fname))

        # Extract embedding
        model = self._registry.get_model()
        model.eval()
        image_np = np.array(cropped)
        augmented = self._transform(image=image_np)
        tensor = augmented["image"].unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = model(tensor).cpu().numpy()

        return embedding
