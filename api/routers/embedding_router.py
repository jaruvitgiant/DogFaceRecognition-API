"""
api/routers/embedding_router.py
---------------------------------
Router สำหรับแปลงรูปภาพสุนัขเป็น embedding vector.
Endpoint: POST /embedding-image/

หลักการ SOLID:
  - SRP: Router รับผิดชอบแค่ HTTP validation และ response formatting
  - DIP: พึ่งพา EmbeddingService ผ่าน Depends()
"""

from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends

from services.embedding_service import EmbeddingService
from api.dependencies import get_embedding_service

router = APIRouter(tags=["Embedding"])


@router.post("/embedding-image/", summary="แปลงรูปภาพสุนัขเป็น Embedding Vector")
async def embedding_image(
    dog_id: int = Form(...),
    files: List[UploadFile] = File(...),
    service: EmbeddingService = Depends(get_embedding_service),
):
    """รับรูปภาพหลายรูป คืน embedding vector สำหรับแต่ละรูป"""
    try:
        results = []
        for file in files:
            result = await service.process_upload_file(file)
            results.append(result)

        return {
            "dog_id": dog_id,
            "processed": len(results),
            "results": results,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
