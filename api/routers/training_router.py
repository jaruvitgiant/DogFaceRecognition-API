"""
api/routers/training_router.py
--------------------------------
Router สำหรับ fine-tune model ด้วยข้อมูลใหม่.
Endpoints: POST /retrain-model-face, GET /train-progress (SSE)

หลักการ SOLID:
  - SRP: Router จัดการแค่ HTTP input/output และ file temp storage
  - DIP: พึ่งพา TrainingOrchestrator ผ่าน Depends()
"""

import asyncio
import os
import shutil
import tempfile
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import json

from services.training_service import TrainingOrchestrator
from infrastructure.auth_service import verify_token_dep
from api.dependencies import get_training_orchestrator

router = APIRouter(tags=["Training"])


@router.get("/train-progress", summary="Stream training progress ผ่าน SSE")
async def stream_training_progress(
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator),
):
    """Server-Sent Events endpoint สำหรับ subscribe training log"""
    queue: asyncio.Queue = asyncio.Queue()
    orchestrator.broadcaster.add_queue(queue)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'status': '📡 เชื่อมต่อกับ Server สำเร็จ'})}\n\n"
            while True:
                data = await queue.get()
                yield data
        except asyncio.CancelledError:
            orchestrator.broadcaster.remove_queue(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/retrain-model-face", summary="เริ่มเทรน model ด้วยรูปภาพใหม่")
async def retrain_model(
    labels: List[str] = Form(...),
    model_type: str = Form("resnet"),
    files: List[UploadFile] = File(...),
    payload=Depends(verify_token_dep),
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator),
):
    """
    รับรูปภาพ + labels → บันทึกชั่วคราว → เริ่ม background training
    Protected ด้วย JWT token (scope: auto_retrain)
    """
    loop = asyncio.get_running_loop()

    # 1. บันทึกไฟล์ลง temp directory
    temp_dir = tempfile.mkdtemp(prefix="train_data_")
    training_data = []

    for label, file in zip(labels, files):
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        training_data.append({"image_path": file_path, "label": label})

    # 2. เริ่ม background training
    orchestrator.start_training(training_data, loop)

    return {
        "message": "Files uploaded and training started in background",
        "temp_directory": temp_dir,
        "received_items": len(training_data),
    }
