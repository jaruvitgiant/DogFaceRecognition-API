import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load model (ย้ายมา CPU เพื่อเลี่ยงปัญหา CUDA re-initialize ใน DataLoader)
model_path = '/home/jaruvitgitant/Documents/Project_findmydog/AI/Detect_dog_faceYOLO11/runs/detect/train3/weights/best.pt'  # ปรับเป็น path ที่ถูกต้องของ model ของคุณ
model = YOLO(model_path).to('cpu') 

TARGET_CLASS = "FaceDog"
CONF_THRESHOLD = 0.6

def yoloCrop(image_input, output_folder="results_faceDog"):
    os.makedirs(output_folder, exist_ok=True)
    """
    image_input: รับได้ทั้ง path (str) หรือ PIL Image
    """
    filename = "unknown.jpg"
    
    # 1. จัดการ Input และดึงชื่อไฟล์มาเตรียมไว้บันทึก
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
        img_for_yolo = image_input
        filename = os.path.basename(image_input) # ดึงชื่อไฟล์ เช่น dog01.jpg
    else:
        img = image_input
        img_for_yolo = image_input
        # ถ้าส่งเป็น PIL Image มาตรงๆ อาจจะไม่มีชื่อไฟล์ ให้ใช้ timestamp หรือตั้งชื่อกลางๆ
        import time
        filename = f"image_{int(time.time()*1000)}.jpg"

    # 2. Run YOLO
    results = model(img_for_yolo, verbose=False, device='cpu')
    
    best_box = None
    best_conf = 0.0
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls_name == TARGET_CLASS and conf >= CONF_THRESHOLD:
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

    # 3. ถ้าเจอ Box ให้ทำการ Crop และบันทึก
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cropped = img.crop((x1, y1, x2, y2))
        
        # บันทึกภาพลง Folder
        
        save_path = os.path.join(output_folder, f"crop_{filename}")
        cropped.save(save_path)
        print("Current Folder:", os.getcwd())
        print("Full path of results:", os.path.abspath("results_faceDog"))
        
        return cropped
    
    # ถ้าไม่เจอเลย ให้ return None หรือส่งภาพเดิมกลับไป
    return None