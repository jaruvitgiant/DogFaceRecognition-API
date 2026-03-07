"""
services/training_service.py
------------------------------
Business logic สำหรับ pipeline การเทรน model ใหม่.

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่ orchestration ของ training workflow
  - OCP: ขยายรองรับ model type ใหม่ได้โดยไม่แก้ core
  - DIP: พึ่งพา IImageProcessor interface ไม่ใช่ YOLO โดยตรง
"""

import asyncio
import json
import os
import threading
import torch
from datetime import datetime
from typing import List, Callable, Optional

from sklearn.preprocessing import LabelEncoder

from core.interfaces import IImageProcessor
from core.config import settings
from resnet.resnet import ResNetBackbone, Bottleneck
from resnet.train import FaceModelTrainer
from resnet.DataLoader import get_dataloaders


def _get_next_version(base_dir: str) -> str:
    """
    หา version ถัดไปใน checkpoint directory
    เช่น v001, v002, ...
    """
    os.makedirs(base_dir, exist_ok=True)
    versions = [
        d for d in os.listdir(base_dir)
        if d.startswith("v") and d[1:].isdigit()
    ]
    if not versions:
        return "v001"
    latest = max(int(v[1:]) for v in versions)
    return f"v{latest + 1:03d}"


class SSEBroadcaster:
    """
    จัดการ Server-Sent Event queues สำหรับ streaming training progress
    แยกออกมาเป็น class ตาม SRP
    """

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    def add_queue(self, queue: asyncio.Queue) -> None:
        self._queues.add(queue)

    def remove_queue(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    async def broadcast(self, data) -> None:
        """ส่งข้อมูลเข้า queue ของ SSE clients ทุกอัน"""
        if not self._queues:
            return
        if isinstance(data, str):
            data = {"status": data}
        formatted = f"data: {json.dumps(data)}\n\n"
        await asyncio.gather(*[q.put(formatted) for q in self._queues])


class TrainingOrchestrator:
    """
    จัดการ background training thread และ SSE progress broadcast
    """

    def __init__(self, image_processor: IImageProcessor) -> None:
        self._processor = image_processor
        self.broadcaster = SSEBroadcaster()
        self._device = settings.DEVICE

        # สร้าง model instance สำหรับเทรน (แยกจาก inference model)
        self._train_model = ResNetBackbone(
            Bottleneck,
            [3, 8, 36, 3],
            embedding_size=settings.EMBEDDING_SIZE,
        ).to(self._device)

    def start_training(
        self,
        training_data: List[dict],
        loop: asyncio.AbstractEventLoop,
    ) -> threading.Thread:
        """
        เริ่ม background training thread
        คืน thread object (เพื่อให้ caller รู้ว่า thread ถูก start แล้ว)
        """
        thread = threading.Thread(
            target=self._run_training,
            args=(training_data, loop),
            daemon=True,
        )
        thread.start()
        return thread

    # ───────────────────────────────────────────────
    # Private: training logic (runs in background thread)
    # ───────────────────────────────────────────────

    def _run_training(
        self,
        training_data: List[dict],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Training pipeline ที่ทำงานใน background thread"""

        def sse_callback(progress_data):
            asyncio.run_coroutine_threadsafe(
                self.broadcaster.broadcast(progress_data), loop
            )

        try:
            if not training_data:
                sse_callback("⚠️ ไม่พบข้อมูลรูปภาพที่ส่งมา")
                return

            sse_callback(f"📦 ได้รับข้อมูล {len(training_data)} รายการ เตรียม Crop...")

            # 1. YOLO Crop images
            cropped_imgs, labels = [], []
            for item in training_data:
                path = item["image_path"]
                if not os.path.exists(path):
                    print(f"File not found: {path}")
                    continue

                result = self._processor.process(path)
                if result is None:
                    continue

                cropped_imgs.append(result)
                labels.append(item["label"])

            if not cropped_imgs:
                sse_callback("❌ ไม่มีรูปภาพที่ใช้ได้หลัง Crop")
                return

            # 2. Label encoding
            sse_callback("กำลังจัดเตรียม Labels...")
            le = LabelEncoder()
            encoded_labels = le.fit_transform(labels)
            actual_num_classes = len(le.classes_)

            # 3. DataLoader
            train_loader, _ = get_dataloaders(
                train_path=cropped_imgs,
                image_ids=encoded_labels,
                batch_size=min(32, len(cropped_imgs)),
            )

            # 4. Versioning
            base_dir = settings.BASE_CHECKPOINT_DIR
            version = _get_next_version(base_dir)
            save_path = os.path.join(base_dir, version)
            os.makedirs(save_path, exist_ok=True)

            # 5. Train
            sse_callback(f"🚀 เริ่มเทรน {version} ({actual_num_classes} classes)...")
            trainer = FaceModelTrainer(
                model=self._train_model,
                train_loader=train_loader,
                device=self._device,
                num_classes=actual_num_classes,
                embedding_size=settings.EMBEDDING_SIZE,
            )
            trainer.train(epochs=3, save_path=save_path, progress_callback=sse_callback)

            # 6. Save weights + metadata
            torch.save(self._train_model.state_dict(), os.path.join(save_path, "model.pth"))
            meta = {
                "model": "resnet152",
                "version": version,
                "trained_at": datetime.now().isoformat(),
                "num_classes": int(actual_num_classes),
                "classes": [int(c) for c in le.classes_],
                "num_images": len(cropped_imgs),
                "epochs": 3,
                "device": str(self._device),
            }
            with open(os.path.join(save_path, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            sse_callback(f"✅ การฝึกสอน {version} เสร็จสมบูรณ์!")

        except Exception as exc:
            sse_callback(f"❌ Error: {str(exc)}")
            print(f"Training error: {exc}")
