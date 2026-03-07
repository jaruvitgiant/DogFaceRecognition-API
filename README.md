SOLID + OOP Refactor — Walkthrough
ผลลัพธ์
app_ubuntu.py
 (730 บรรทัด, God Object) → Layered Architecture (20 ไฟล์ใหม่)

โครงสร้างใหม่:

api-modelfaceNetDog/
├── core/
│   ├── interfaces.py      ← Abstract Base Classes
│   └── config.py          ← Pydantic Settings (โหลดจาก .env)
├── infrastructure/
│   ├── model_registry.py  ← ModelRegistry (OCP + SRP)
│   ├── image_processor.py ← YoloCropProcessor (SRP, lazy load)
│   └── auth_service.py    ← JWTAuthService (SRP)
├── services/
│   ├── embedding_service.py    ← EmbeddingService
│   ├── search_service.py       ← KNNSearchService
│   ├── training_service.py     ← TrainingOrchestrator + SSEBroadcaster
│   └── visualization_service.py ← VisualizationService
├── api/
│   ├── dependencies.py         ← DI Container (singleton instances)
│   └── routers/
│       ├── model_router.py     ← /model/models, /model/current-model, /model/select-model
│       ├── embedding_router.py ← /embedding-image/
│       ├── search_router.py    ← /SEARCH-DOG02/
│       ├── training_router.py  ← /retrain-model-face, /train-progress
│       └── knn_router.py       ← /tiger_knnTrain/, /test-knn/
├── app.py                      ← 🆕 Entry point ใหม่
└── app_ubuntu.py               ← เก็บไว้เป็น backup
SOLID Mapping
หลักการ	Implementation
S Single Responsibility	ทุก class/module รับผิดชอบแค่เรื่องเดียว
O Open/Closed	ModelRegistry.list_checkpoints() เพิ่ม type ใหม่ได้โดยไม่แก้ router
L Liskov Substitution	
YoloCropProcessor
 แทน 
IImageProcessor
 ได้สมบูรณ์
I Interface Segregation	แยก interface: 
IImageProcessor
, 
IEmbeddingModel
, 
ISearchService
, 
IModelRegistry
, 
IAuthService
D Dependency Inversion	Router พึ่ง interface ผ่าน Depends() ไม่ hard-code concrete class
สิ่งที่เปลี่ยน
ก่อน (app_ubuntu.py)
Global variables: model152, ACTIVE_MODEL_NAME, device
Import ซ้ำซ้อน (FastAPI import 3 ครั้ง)
Business logic ในข้างใน route function โดยตรง
ไม่มี abstraction ระหว่าง layer
หลัง (Layered Architecture)
State encapsulated ใน 
ModelRegistry
 class
Services inject dependency ผ่าน constructor
Routers เป็นแค่ HTTP layer얇 — delegate ทุกอย่างไปหา service
YoloCropProcessor
 ใช้ lazy loading (ลด startup time)
SSEBroadcaster
 แยกออกมาจาก training logic (SRP)
Config centralised ใน 
Settings
 class (ใช้ .env)
Syntax Check Result
✅ All 16 files — syntax OK (py_compile)
Files checked: 
core/interfaces.py
, 
core/config.py
, infrastructure/*.py, services/*.py, 
api/dependencies.py
, api/routers/*.py, 
app.py

วิธีรันระบบใหม่
bash
# รัน development server ด้วย app.py ใหม่
uvicorn app:app --reload --port 8001
# หรือผ่าน Docker (ถ้า Dockerfile ใช้ app_ubuntu กับ uvicorn)
# แก้ CMD ใน Dockerfile จาก:
#   CMD ["uvicorn", "app_ubuntu:app", ...]
# เป็น:
#   CMD ["uvicorn", "app:app", ...]
IMPORTANT

Model routers ย้าย prefix — endpoints กลุ่ม model management มี prefix /model แล้ว:

/models → /model/models
/current-model → /model/current-model
/select-model → /model/select-model
endpoints อื่นๆ (/SEARCH-DOG02/, /embedding-image/, /retrain-model-face, /train-progress, /tiger_knnTrain/, /test-knn/) ยังคงเดิม

NOTE

app_ubuntu.py ยังคงอยู่ในโปรเจกต์เป็น backup — ไม่ได้ถูกลบหรือแก้ไข

uvicorn app:app --reload --port 8001