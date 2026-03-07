"""
core/interfaces.py
------------------
Abstract Base Classes ที่กำหนด CONTRACT ของแต่ละ component ใน system.

หลักการ SOLID ที่ใช้:
  - ISP (Interface Segregation Principle): แยก interface ตาม responsibility
  - DIP (Dependency Inversion Principle): High-level modules พึ่งพา abstractions นี้
  - LSP (Liskov Substitution Principle): Subclass ต้องใช้แทนกันได้
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image


# ───────────────────────────────────────────────────────────────
# Interface: Image Processing
# ───────────────────────────────────────────────────────────────

class IImageProcessor(ABC):
    """
    สัญญาสำหรับ component ที่รับผิดชอบประมวลผลภาพ (เช่น crop ใบหน้าสุนัข)
    """

    @abstractmethod
    def process(self, image_input) -> Optional[Image.Image]:
        """
        รับ input เป็น path (str) หรือ PIL Image
        คืนค่า PIL Image ที่ crop แล้ว หรือ None ถ้าไม่พบ
        """
        ...


# ───────────────────────────────────────────────────────────────
# Interface: Embedding Model
# ───────────────────────────────────────────────────────────────

class IEmbeddingModel(ABC):
    """
    สัญญาสำหรับ Neural Network ที่แปลงภาพเป็น embedding vector
    """

    @abstractmethod
    def get_embedding(self, img_pil: Image.Image) -> np.ndarray:
        """
        รับ PIL Image คืนค่า 1D numpy array (embedding vector)
        """
        ...


# ───────────────────────────────────────────────────────────────
# Interface: Model Registry
# ───────────────────────────────────────────────────────────────

class IModelRegistry(ABC):
    """
    สัญญาสำหรับ component ที่จัดการ lifecycle ของ model weights
    (โหลด, switch, list versions)
    """

    @abstractmethod
    def load(self, model_path: str) -> bool:
        """โหลด weight จาก path — คืน True ถ้าสำเร็จ"""
        ...

    @abstractmethod
    def get_model(self):
        """คืน pytorch model ที่โหลดอยู่ (หรือ None)"""
        ...

    @abstractmethod
    def get_active_name(self) -> Optional[str]:
        """คืนชื่อ/version ของ model ที่ active อยู่"""
        ...

    @abstractmethod
    def get_active_path(self) -> Optional[str]:
        """คืน path ของ model ที่ active อยู่"""
        ...


# ───────────────────────────────────────────────────────────────
# Interface: Search Service
# ───────────────────────────────────────────────────────────────

class ISearchService(ABC):
    """
    สัญญาสำหรับ component ที่ทำ nearest-neighbor search
    (เปลี่ยน algorithm ได้โดยไม่กระทบ router)
    """

    @abstractmethod
    def train(self, embeddings: List[np.ndarray], labels: List) -> None:
        """เทรน index จาก embedding + labels"""
        ...

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        ค้นหา top_k รายการที่ใกล้เคียงที่สุด
        คืน list ของ {"dog_id": ..., "distance": ...}
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """บันทึก model ลง disk"""
        ...

    @abstractmethod
    def load(self) -> bool:
        """โหลด model จาก disk — คืน True ถ้าสำเร็จ"""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """ตรวจสอบว่า model พร้อมใช้งานหรือยัง"""
        ...


# ───────────────────────────────────────────────────────────────
# Interface: Auth Service
# ───────────────────────────────────────────────────────────────

class IAuthService(ABC):
    """
    สัญญาสำหรับ authentication/authorization
    """

    @abstractmethod
    def verify(self, token: str) -> dict:
        """
        ตรวจสอบ token — คืน payload dict ถ้า valid
        raise HTTPException ถ้าไม่ valid
        """
        ...
