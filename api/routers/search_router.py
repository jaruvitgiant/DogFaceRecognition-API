"""
api/routers/search_router.py
------------------------------
Router สำหรับค้นหาสุนัขโดยใช้ KNN.
Endpoint: POST /SEARCH-DOG02/

หลักการ SOLID:
  - SRP: Router รับผิดชอบแค่ HTTP layer
  - DIP: พึ่งพา ISearchService ผ่าน Depends()
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from services.search_service import KNNSearchService
from api.dependencies import get_search_service

router = APIRouter(tags=["Dog Search"])


@router.post("/SEARCH-DOG02/", summary="ค้นหาสุนัขจากรูปภาพ (KNN)")
async def search_dog(
    file: UploadFile = File(...),
    service: KNNSearchService = Depends(get_search_service),
):
    """
    รับรูปภาพสุนัข → YOLO crop → extract embedding → KNN search
    คืน top-5 สุนัขที่ใกล้เคียงที่สุด
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=400,
            detail="KNN model not trained yet. Please call /tiger_knnTrain/ first.",
        )

    # Extract embedding จากภาพที่อัปโหลด
    embedding = await service.extract_embedding_from_upload(file)

    if embedding is None:
        return {
            "status": "not_found",
            "message": "ไม่พบหมาในระบบ (YOLO ตรวจไม่พบสุนัขในภาพ)",
            "results": [],
        }

    results = service.search(embedding, top_k=5)
    return {"results": results}
