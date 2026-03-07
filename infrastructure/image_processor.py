"""
infrastructure/image_processor.py
-----------------------------------
Concrete implementation ของ IImageProcessor โดยใช้ YOLO.

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่การ detect และ crop dog face จากภาพ
  - OCP: ถ้าอยากเปลี่ยนเป็น model อื่น (เช่น MediaPipe) ให้ extend class ใหม่
  - DIP: Implements IImageProcessor — caller พึ่ง interface ไม่ใช่ YOLO โดยตรง
"""

import os
import time
from typing import Optional

from PIL import Image

from core.interfaces import IImageProcessor
from core.config import settings


class YoloCropProcessor(IImageProcessor):
    """
    ใช้ YOLO11 ตรวจจับใบหน้าสุนัข แล้ว crop ออกมา
    
    Lazy-loading: YOLO model จะถูกโหลดเมื่อมีการเรียกใช้ครั้งแรก
    ไม่ใช่ตอน import module (ลดเวลา startup และ memory footprint)
    """

    def __init__(
        self,
        model_path: str = settings.YOLO_MODEL_PATH,
        target_class: str = settings.YOLO_TARGET_CLASS,
        conf_threshold: float = settings.YOLO_CONF_THRESHOLD,
        output_folder: str = "results_faceDog",
    ) -> None:
        self._model_path = model_path
        self._target_class = target_class
        self._conf_threshold = conf_threshold
        self._output_folder = output_folder
        self._yolo = None  # Lazy load

    def _ensure_model_loaded(self) -> None:
        """โหลด YOLO model ถ้ายังไม่ได้โหลด (Lazy initialization)"""
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self._model_path).to("cpu")

    # ───────────────────────────────────────────────
    # IImageProcessor contract
    # ───────────────────────────────────────────────

    def process(self, image_input) -> Optional[Image.Image]:
        """
        รับ image_input เป็น:
          - str: file path
          - PIL.Image.Image: image object

        คืนค่า PIL Image ที่ crop เฉพาะใบหน้าสุนัข
        หรือ None ถ้า YOLO ตรวจไม่เจอ
        """
        self._ensure_model_loaded()
        os.makedirs(self._output_folder, exist_ok=True)

        # 1. Normalize input → PIL Image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
            img_for_yolo = image_input
            filename = os.path.basename(image_input)
        else:
            img = image_input.convert("RGB") if hasattr(image_input, "convert") else image_input
            img_for_yolo = image_input
            filename = f"image_{int(time.time() * 1000)}.jpg"

        # 2. Run YOLO detection
        results = self._yolo(img_for_yolo, verbose=False, device="cpu")

        # 3. หา bounding box ที่ดีที่สุด (confidence สูงสุด)
        best_box = None
        best_conf = 0.0

        for result in results:
            for box in result.boxes:
                cls_name = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls_name == self._target_class and conf >= self._conf_threshold:
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

        # 4. Crop และบันทึกถ้าเจอ
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cropped = img.crop((x1, y1, x2, y2))

            save_path = os.path.join(self._output_folder, f"crop_{filename}")
            cropped.save(save_path)

            return cropped

        return None
