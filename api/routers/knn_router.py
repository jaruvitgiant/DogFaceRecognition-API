"""
api/routers/knn_router.py
---------------------------
Router สำหรับ KNN training และ visualization.
Endpoints: POST /tiger_knnTrain/, POST /test-knn/

หลักการ SOLID:
  - SRP: Router รับผิดชอบแค่ HTTP parsing และ formatting response
  - DIP: พึ่งพา KNNSearchService และ VisualizationService ผ่าน Depends()
"""

import base64
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from services.search_service import KNNSearchService
from services.visualization_service import VisualizationService
from api.dependencies import get_search_service, get_visualization_service
from infrastructure.model_registry import ModelRegistry
from api.dependencies import get_model_registry

router = APIRouter(tags=["KNN Classifier"])


# ── Pydantic Request Schemas ──────────────────────────────────────────

class EmbeddingItem(BaseModel):
    dog_id: int
    embedding_b64: str


class KNNRequest(BaseModel):
    data: List[EmbeddingItem]


# ── Helper ───────────────────────────────────────────────────────────

def _decode_embeddings(items: List[EmbeddingItem]) -> tuple[List[np.ndarray], List[int]]:
    """แปลง base64 embedding items เป็น numpy arrays"""
    embeddings, labels = [], []
    for item in items:
        try:
            binary = base64.b64decode(item.embedding_b64)
            emb = np.frombuffer(binary, dtype=np.float32)
            if emb.ndim != 1 or emb.shape[0] == 0:
                continue
            embeddings.append(emb)
            labels.append(item.dog_id)
        except Exception as exc:
            print(f"Error decoding dog_id {item.dog_id}: {exc}")
    return embeddings, labels


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/tiger_knnTrain/", summary="เทรน KNN index จาก embeddings")
async def train_knn(
    request: KNNRequest,
    service: KNNSearchService = Depends(get_search_service),
):
    """รับ list ของ embedding vectors → เทรน KNN → บันทึกลง disk"""
    embeddings, labels = _decode_embeddings(request.data)

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid embeddings provided")

    service.train(embeddings, labels)
    service.save()

    return {
        "status": "success",
        "total_embeddings_trained": len(embeddings),
    }


@router.post("/test-knn/", summary="ทดสอบ KNN และแสดง t-SNE / Confusion Matrix")
async def test_knn(
    request: KNNRequest,
    search_service: KNNSearchService = Depends(get_search_service),
    viz_service: VisualizationService = Depends(get_visualization_service),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    รับ embeddings → สร้าง t-SNE plot + KNN confusion matrix
    คืน base64 images พร้อม accuracy score
    """
    try:
        embeddings, labels = _decode_embeddings(request.data)

        if not embeddings:
            raise HTTPException(status_code=400, detail="No valid embeddings")

        X = np.array(embeddings)
        y = np.array(labels)

        if len(X) < 2:
            raise ValueError("ต้องมีข้อมูลอย่างน้อย 2 จุดสำหรับ visualization")

        # t-SNE plot
        tsne_image = viz_service.create_tsne_plot(X, y)
 
        # Confusion matrix + accuracy
        cm_image, accuracy = viz_service.create_confusion_matrix(X, y)

        return {
            "status": "success",
            "accuracy": float(accuracy),
            "tsne_plot": tsne_image,
            "knn_matrix": cm_image,
            "model_name": registry.get_active_name(),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
