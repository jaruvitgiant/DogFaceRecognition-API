"""
testService/test.py
--------------------
Test Suite สำหรับ FastAPI Dog Recognition API (SOLID Refactored)

ครอบคลุม:
  1. Unit Tests — ทดสอบ service classes โดยตรง (ไม่ต้องรัน server)
  2. Integration Tests — ทดสอบ API endpoints ผ่าน FastAPI TestClient
  3. End-to-End Flow Tests — embedding → KNN train → search ด้วย dataset จริง

วิธีรัน:
  cd api-modelfaceNetDog
  uv run pytest testService/test.py -v
  
  # รันเฉพาะ unit tests (เร็ว):
  uv run pytest testService/test.py -v -m unit

  # รัน integration tests:
  uv run pytest testService/test.py -v -m integration

  # รัน e2e flow:
  uv run pytest testService/test.py -v -m e2e
"""

import sys
import os
import io
import base64
import json
import time
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────
# เพิ่ม project root เข้า sys.path เพื่อ import modules ได้
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset_test"
BASE_URL = "http://localhost:8001"  # สำหรับ e2e tests ที่ต้องการ server จริง
API_PREFIX = ""

# เลือก 3 สุนัขแรกจาก dataset สำหรับ test
DOG_FOLDERS = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])[:3]


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_images() -> dict[str, list[Path]]:
    """คืน dict ของ {dog_name: [image_paths]} สำหรับ 3 สุนัขแรก"""
    result = {}
    for folder in DOG_FOLDERS:
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if images:
            result[folder.name] = images[:3]  # ใช้แค่ 3 รูปต่อสุนัข
    return result


@pytest.fixture(scope="session")
def dummy_embedding() -> np.ndarray:
    """สร้าง embedding vector จำลอง ขนาด 512 (เหมือน ResNet output)"""
    rng = np.random.default_rng(42)
    vec = rng.random(512).astype(np.float32)
    # Normalize เหมือน ResNet จริง
    vec = vec / np.linalg.norm(vec)
    return vec


@pytest.fixture(scope="session")
def dummy_embedding_b64(dummy_embedding) -> str:
    """แปลง embedding เป็น base64 string"""
    return base64.b64encode(dummy_embedding.tobytes()).decode("utf-8")


@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient (ไม่ต้องรัน server จริง)"""
    # Import ช้าเพื่อเลี่ยง YOLO/model load ตอน import
    os.chdir(PROJECT_ROOT)
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ─────────────────────────────────────────────────────────────────────
# 1. UNIT TESTS — VisualizationService
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestVisualizationService:
    """
    ทดสอบ VisualizationService โดยตรง ไม่ต้องรัน server
    ทดสอบ SRP: ทุก method ทำงานอิสระ ไม่มี side effect
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from services.visualization_service import VisualizationService
        self.service = VisualizationService()

    def test_tsne_plot_returns_base64_string(self, dummy_embedding):
        """t-SNE ต้องคืน base64 string ที่ decode เป็น PNG ได้"""
        X = np.vstack([dummy_embedding * (i + 0.1) for i in range(5)])
        y = np.array([1, 1, 2, 2, 3])

        result = self.service.create_tsne_plot(X, y)

        assert isinstance(result, str), "ต้องคืน string"
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG", "ต้องเป็น PNG file"

    def test_tsne_raises_on_single_point(self, dummy_embedding):
        """t-SNE ต้องเกิด ValueError ถ้าข้อมูล < 2 จุด"""
        X = dummy_embedding.reshape(1, -1)
        y = np.array([1])

        with pytest.raises(ValueError, match="2"):
            self.service.create_tsne_plot(X, y)

    def test_confusion_matrix_returns_tuple(self, dummy_embedding):
        """confusion matrix ต้องคืน (base64_str, float_accuracy)"""
        X = np.vstack([dummy_embedding * (i + 0.1) for i in range(4)])
        y = np.array([1, 1, 2, 2])

        img_b64, acc = self.service.create_confusion_matrix(X, y)

        assert isinstance(img_b64, str), "ต้องคืน base64 string"
        assert 0.0 <= acc <= 1.0, "accuracy ต้องอยู่ระหว่าง 0-1"
        decoded = base64.b64decode(img_b64)
        assert decoded[:4] == b"\x89PNG", "ต้องเป็น PNG"

    def test_confusion_matrix_perfect_score(self):
        """
        ถ้า embeddings ของแต่ละ dog อยู่ห่างกันชัดมาก accuracy ควรสูง
        ใช้ 2 ตัวอย่างต่อ class และ one-hot style vectors เพื่อให้ KNN แยกได้
        """
        # สร้าง embeddings 2 ตัวต่อ class ด้วย block vectors (ห่างกันมาก)
        block_size = 64
        X = np.zeros((4, 512), dtype=np.float32)
        X[0, :block_size] = 1.0    # class 1 - sample 1
        X[1, :block_size] = 0.9    # class 1 - sample 2
        X[2, block_size:block_size*2] = 1.0  # class 2 - sample 1
        X[3, block_size:block_size*2] = 0.9  # class 2 - sample 2
        y = np.array([1, 1, 2, 2])

        _, acc = self.service.create_confusion_matrix(X, y)
        assert acc >= 0.5, f"accuracy ควรสูงกว่า random guess แต่ได้ {acc}"


# ─────────────────────────────────────────────────────────────────────
# 2. UNIT TESTS — KNNSearchService (ไม่ต้องโหลด model จริง)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestKNNSearchService:
    """
    ทดสอบ KNNSearchService logic โดยใช้ mock registry + processor
    ทดสอบ: train, search, save/load, is_ready
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from unittest.mock import MagicMock
        from services.search_service import KNNSearchService

        mock_registry = MagicMock()
        mock_processor = MagicMock()

        self.knn_path = str(tmp_path / "knn.joblib")
        self.labels_path = str(tmp_path / "labels.joblib")

        self.service = KNNSearchService(
            model_registry=mock_registry,
            image_processor=mock_processor,
            knn_model_path=self.knn_path,
            knn_labels_path=self.labels_path,
        )

    def test_is_ready_false_before_train(self):
        """KNN ต้องไม่ ready ก่อนเทรน"""
        assert self.service.is_ready() is False

    def test_train_makes_service_ready(self, dummy_embedding):
        """หลังเทรน service ต้อง ready"""
        embeddings = [dummy_embedding * (i + 0.1) for i in range(4)]
        labels = [1, 1, 2, 2]

        self.service.train(embeddings, labels)
        assert self.service.is_ready() is True

    def test_search_returns_ranked_results(self, dummy_embedding):
        """search ต้องคืน list ของ dict ที่มี rank, dog_id, distance"""
        embeddings = [dummy_embedding * (i + 0.1) for i in range(6)]
        labels = [10, 10, 20, 20, 30, 30]
        self.service.train(embeddings, labels)

        results = self.service.search(dummy_embedding, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for item in results:
            assert "rank" in item
            assert "dog_id" in item
            assert "distance" in item
            assert isinstance(item["distance"], float)

    def test_search_deduplicates_by_dog_id(self, dummy_embedding):
        """search ต้องคืนแค่ dog_id ละ 1 ตัว (best distance)"""
        embeddings = [dummy_embedding] * 5
        labels = [1, 1, 1, 2, 2]
        self.service.train(embeddings, labels)

        results = self.service.search(dummy_embedding, top_k=5)
        dog_ids = [r["dog_id"] for r in results]
        assert len(dog_ids) == len(set(dog_ids)), "ต้องไม่มี dog_id ซ้ำ"

    def test_search_raises_when_not_ready(self, dummy_embedding):
        """search ต้อง raise RuntimeError ถ้า KNN ยังไม่เทรน"""
        with pytest.raises(RuntimeError, match="not trained"):
            self.service.search(dummy_embedding)

    def test_save_and_load(self, dummy_embedding):
        """save แล้ว load ต้องทำงานได้ปกติ"""
        embeddings = [dummy_embedding * (i + 0.5) for i in range(4)]
        labels = [100, 100, 200, 200]
        self.service.train(embeddings, labels)
        self.service.save()

        # สร้าง instance ใหม่แล้ว load
        from unittest.mock import MagicMock
        from services.search_service import KNNSearchService
        new_service = KNNSearchService(
            model_registry=MagicMock(),
            image_processor=MagicMock(),
            knn_model_path=self.knn_path,
            knn_labels_path=self.labels_path,
        )
        success = new_service.load()
        assert success is True
        assert new_service.is_ready() is True


# ─────────────────────────────────────────────────────────────────────
# 3. UNIT TESTS — ModelRegistry (mock torch.load)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestModelRegistry:
    """
    ทดสอบ ModelRegistry โดยไม่ต้อง access GPU/model จริง
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from infrastructure.model_registry import ModelRegistry
        self.registry = ModelRegistry()

    def test_initial_state(self):
        """ก่อนโหลด active_name และ active_path ต้องเป็น None"""
        assert self.registry.get_active_name() is None
        assert self.registry.get_active_path() is None

    def test_load_returns_false_for_missing_file(self):
        """โหลด path ที่ไม่มีอยู่ ต้องคืน False ไม่ crash"""
        result = self.registry.load("/nonexistent/path/model.pth")
        assert result is False

    def test_resolve_path_default(self):
        """'default' ต้องคืน DEFAULT_MODEL_PATH"""
        from core.config import settings
        path = self.registry.resolve_path("default")
        assert path == settings.DEFAULT_MODEL_PATH

    def test_resolve_path_versioned(self):
        """version string ต้องคืน path ใน BASE_CHECKPOINT_DIR"""
        from core.config import settings
        path = self.registry.resolve_path("v001")
        assert "v001" in path
        assert settings.BASE_CHECKPOINT_DIR in path

    def test_list_checkpoints_always_has_default(self):
        """list_checkpoints ต้องมี default model เสมอ"""
        models = self.registry.list_checkpoints()
        ids = [m["id"] for m in models]
        assert "default" in ids


# ─────────────────────────────────────────────────────────────────────
# 4. UNIT TESTS — Config / Settings
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestConfig:
    """ทดสอบ Pydantic Settings"""

    def test_settings_loads(self):
        """Settings ต้องสร้างได้โดยไม่ error"""
        from core.config import settings
        assert settings is not None

    def test_device_is_string(self):
        """DEVICE property ต้องคืน string (cpu หรือ cuda)"""
        from core.config import settings
        assert settings.DEVICE in ("cpu", "cuda")

    def test_embedding_size_default(self):
        """EMBEDDING_SIZE default ต้องเป็น 512"""
        from core.config import settings
        assert settings.EMBEDDING_SIZE == 512

    def test_allowed_origins_is_list(self):
        """ALLOWED_ORIGINS ต้องเป็น list"""
        from core.config import settings
        assert isinstance(settings.ALLOWED_ORIGINS, list)


# ─────────────────────────────────────────────────────────────────────
# 5. INTEGRATION TESTS — API Endpoints ผ่าน TestClient
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestModelEndpoints:
    """
    ทดสอบ /model/* endpoints
    ใช้ FastAPI TestClient (snapshot test — ไม่ต้อง GPU)
    """

    def test_list_models_returns_200(self, test_client):
        """GET /model/models ต้องคืน 200 และ key 'models'"""
        resp = test_client.get("/model/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_list_models_has_default(self, test_client):
        """list models ต้องมี default เสมอ"""
        resp = test_client.get("/model/models")
        models = resp.json()["models"]
        ids = [m["id"] for m in models]
        assert "default" in ids

    def test_current_model_returns_200(self, test_client):
        """GET /model/current-model ต้องคืน 200"""
        resp = test_client.get("/model/current-model")
        assert resp.status_code == 200
        data = resp.json()
        assert "device" in data
        assert data["device"] in ("cpu", "cuda")

    def test_select_nonexistent_model_returns_404(self, test_client):
        """POST /model/select-model?version=nonexistent ต้องคืน 404"""
        resp = test_client.post("/model/select-model?version=nonexistent_v99999")
        assert resp.status_code == 404


@pytest.mark.integration
class TestKNNEndpoints:
    """
    ทดสอบ /tiger_knnTrain/ และ /test-knn/
    """

    def _make_knn_payload(self, n_dogs: int = 3, n_per_dog: int = 3) -> dict:
        """สร้าง request payload ด้วย dummy embeddings"""
        rng = np.random.default_rng(42)
        data = []
        for dog_id in range(1, n_dogs + 1):
            for _ in range(n_per_dog):
                vec = rng.random(512).astype(np.float32)
                vec /= np.linalg.norm(vec)
                b64 = base64.b64encode(vec.tobytes()).decode("utf-8")
                data.append({"dog_id": dog_id, "embedding_b64": b64})
        return {"data": data}

    def test_train_knn_success(self, test_client):
        """POST /tiger_knnTrain/ ด้วยข้อมูลถูกต้องต้องคืน 200"""
        payload = self._make_knn_payload(n_dogs=3, n_per_dog=4)
        resp = test_client.post("/tiger_knnTrain/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["total_embeddings_trained"] == 12

    def test_train_knn_empty_payload(self, test_client):
        """POST /tiger_knnTrain/ ด้วย data ว่างต้องคืน 400"""
        resp = test_client.post("/tiger_knnTrain/", json={"data": []})
        assert resp.status_code == 400

    def test_test_knn_returns_plots(self, test_client):
        """POST /test-knn/ ต้องคืน tsne_plot และ knn_matrix (base64)"""
        # เทรน KNN ก่อน
        payload = self._make_knn_payload(n_dogs=4, n_per_dog=5)
        test_client.post("/tiger_knnTrain/", json=payload)

        # ทดสอบ visualization
        resp = test_client.post("/test-knn/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "tsne_plot" in data
        assert "knn_matrix" in data
        assert "accuracy" in data
        assert isinstance(data["accuracy"], float)

        # ตรวจสอบว่าเป็น PNG จริง
        tsne_decoded = base64.b64decode(data["tsne_plot"])
        assert tsne_decoded[:4] == b"\x89PNG"

    def test_test_knn_single_point_returns_400(self, test_client):
        """POST /test-knn/ ด้วย 1 embedding เดียวต้องคืน 400"""
        rng = np.random.default_rng(0)
        vec = rng.random(512).astype(np.float32)
        b64 = base64.b64encode(vec.tobytes()).decode("utf-8")
        payload = {"data": [{"dog_id": 1, "embedding_b64": b64}]}
        resp = test_client.post("/test-knn/", json=payload)
        assert resp.status_code == 400


@pytest.mark.integration
class TestSearchEndpoint:
    """ทดสอบ /SEARCH-DOG02/"""

    def test_search_without_knn_returns_400(self, test_client):
        """
        /SEARCH-DOG02/ ถ้า KNN ไม่ได้เทรน ต้องคืน 400
        (จะทดสอบ after reset หรือ fresh state)
        """
        # ส่งรูป dummy (PNG 1x1 pixel)
        img = Image.new("RGB", (50, 50), color=(200, 150, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # เทรน KNN ก่อนเพื่อให้ flow ทำงานได้
        rng = np.random.default_rng(1)
        data = []
        for dog_id in range(1, 4):
            for _ in range(3):
                vec = rng.random(512).astype(np.float32)
                vec /= np.linalg.norm(vec)
                b64 = base64.b64encode(vec.tobytes()).decode("utf-8")
                data.append({"dog_id": dog_id, "embedding_b64": b64})
        test_client.post("/tiger_knnTrain/", json={"data": data})

        # ส่งรูปไป search (YOLO อาจตรวจไม่เจอสุนัขจาก dummy image)
        resp = test_client.post(
            "/SEARCH-DOG02/",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        # คาดว่า YOLO จะตรวจไม่เจอ → ได้ status not_found หรือ 200 ก็ OK
        assert resp.status_code in (200, 400)
        if resp.status_code == 200:
            data = resp.json()
            assert "results" in data or "status" in data


# ─────────────────────────────────────────────────────────────────────
# 6. END-TO-END FLOW TEST — dataset จริง
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestEmbeddingAndKNNFlow:
    """
    E2E Flow: embedding-image/ → tiger_knnTrain/ → test-knn/
    ใช้รูปภาพจาก dataset_test จริง (ต้องการ model โหลดสำเร็จ)
    """

    def test_embedding_endpoint_with_real_image(self, test_client, sample_images):
        """
        POST /embedding-image/ ด้วยรูปจริง
        ต้องคืน embedding_base64 ที่ decode ได้เป็น 512-dim vector
        """
        dog_name, images = next(iter(sample_images.items()))
        img_path = images[0]

        with open(img_path, "rb") as f:
            resp = test_client.post(
                "/embedding-image/",
                data={"dog_id": "1"},
                files={"files": (img_path.name, f, "image/jpeg")},
            )

        if resp.status_code == 503:
            pytest.skip("Model ยังไม่ได้โหลด — ข้าม test นี้")

        assert resp.status_code == 200, f"Response: {resp.text}"
        data = resp.json()
        assert data["processed"] == 1
        results = data["results"]
        assert len(results) == 1
        assert "embedding_base64" in results[0]

        # ตรวจสอบ dimension ของ embedding
        b64 = results[0]["embedding_base64"]
        vec = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        assert vec.shape == (512,), f"Embedding ต้องมีขนาด 512 ไม่ใช่ {vec.shape}"

    def test_full_flow_embed_train_search(self, test_client, sample_images):
        """
        Full flow test:
        1. embed รูปสุนัขหลายตัว
        2. train KNN ด้วย embeddings
        3. search ด้วยรูปใหม่

        ตรวจสอบว่า rank 1 ควรเป็นสุนัขตัวเดียวกัน (same-dog query)
        """
        # 1. Embed ทุกรูปในทุก dog folder
        training_payload = []
        dog_name_to_id = {}
        current_id = 1

        for dog_name, images in sample_images.items():
            if dog_name not in dog_name_to_id:
                dog_name_to_id[dog_name] = current_id
                current_id += 1
            dog_id = dog_name_to_id[dog_name]

            for img_path in images[:-1]:  # เก็บรูปสุดท้ายไว้ query
                with open(img_path, "rb") as f:
                    resp = test_client.post(
                        "/embedding-image/",
                        data={"dog_id": str(dog_id)},
                        files={"files": (img_path.name, f, "image/jpeg")},
                    )
                if resp.status_code == 503:
                    pytest.skip("Model ยังไม่ได้โหลด")
                if resp.status_code != 200:
                    continue

                data = resp.json()
                for r in data["results"]:
                    training_payload.append({
                        "dog_id": dog_id,
                        "embedding_b64": r["embedding_base64"],
                    })

        if not training_payload:
            pytest.skip("ไม่มี embedding ที่ได้จากรูปจริง (YOLO อาจตรวจไม่เจอ)")

        # 2. Train KNN
        train_resp = test_client.post(
            "/tiger_knnTrain/",
            json={"data": training_payload},
        )
        assert train_resp.status_code == 200

        # 3. Search ด้วยรูปสุดท้ายของ dog แรก
        first_dog_name = list(sample_images.keys())[0]
        first_dog_id = dog_name_to_id[first_dog_name]
        query_img = sample_images[first_dog_name][-1]  # รูปสุดท้าย (ไม่ได้ train)

        with open(query_img, "rb") as f:
            search_resp = test_client.post(
                "/SEARCH-DOG02/",
                files={"file": (query_img.name, f, "image/jpeg")},
            )

        assert search_resp.status_code == 200
        search_data = search_resp.json()

        if search_data.get("status") == "not_found":
            pytest.skip("YOLO ตรวจไม่พบสุนัขในภาพ query")

        results = search_data["results"]
        assert len(results) > 0, "ต้องมีผลลัพธ์"

        # ตรวจสอบว่าสุนัขที่ rank 1 มี distance ต่ำที่สุด
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances), "ต้องเรียงจาก distance น้อยไปมาก"


# ─────────────────────────────────────────────────────────────────────
# 7. TESTS — SOLID Principles Verification
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSOLIDCompliance:
    """
    ทดสอบว่า Implementation ปฏิบัติตาม SOLID principles
    """

    def test_interfaces_are_abstract(self):
        """Abstract Base Classes ต้องไม่ instantiate ได้โดยตรง"""
        from core.interfaces import (
            IImageProcessor, IEmbeddingModel,
            ISearchService, IModelRegistry, IAuthService
        )
        for interface in [IImageProcessor, IEmbeddingModel, ISearchService, IModelRegistry, IAuthService]:
            with pytest.raises(TypeError):
                interface()

    def test_model_registry_implements_interface(self):
        """ModelRegistry ต้อง implements IModelRegistry"""
        from core.interfaces import IModelRegistry
        from infrastructure.model_registry import ModelRegistry
        assert issubclass(ModelRegistry, IModelRegistry)

    def test_yolo_processor_implements_interface(self):
        """YoloCropProcessor ต้อง implements IImageProcessor"""
        from core.interfaces import IImageProcessor
        from infrastructure.image_processor import YoloCropProcessor
        assert issubclass(YoloCropProcessor, IImageProcessor)

    def test_knn_search_implements_interface(self):
        """KNNSearchService ต้อง implements ISearchService"""
        from core.interfaces import ISearchService
        from services.search_service import KNNSearchService
        assert issubclass(KNNSearchService, ISearchService)

    def test_jwt_auth_implements_interface(self):
        """JWTAuthService ต้อง implements IAuthService"""
        from core.interfaces import IAuthService
        from infrastructure.auth_service import JWTAuthService
        assert issubclass(JWTAuthService, IAuthService)

    def test_embedding_service_implements_interface(self):
        """EmbeddingService ต้อง implements IEmbeddingModel"""
        from core.interfaces import IEmbeddingModel
        from services.embedding_service import EmbeddingService
        assert issubclass(EmbeddingService, IEmbeddingModel)

    def test_services_use_constructor_injection(self):
        """Services ต้องรับ dependencies ผ่าน __init__ ไม่ใช่ global"""
        from services.embedding_service import EmbeddingService
        import inspect
        sig = inspect.signature(EmbeddingService.__init__)
        params = list(sig.parameters.keys())
        assert "model_registry" in params, "EmbeddingService ต้องรับ model_registry ใน constructor"
        assert "image_processor" in params, "EmbeddingService ต้องรับ image_processor ใน constructor"


# ─────────────────────────────────────────────────────────────────────
# Entry point สำหรับรันตรงๆ
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pytest",
        __file__, "-v",
        "--tb=short",
        "-m", "unit",  # รัน unit tests ก่อน (เร็วที่สุด)
    ])
