import os
import cv2
from ultralytics import YOLO

# --- BASE PATH RELATIVE TO SCRIPT ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder containing this script
IMAGE_FOLDER = os.path.join(BASE_DIR, "Vis")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Results")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- LOAD YOLOv11 MODEL ---
model = YOLO("yolo11n.pt")  # replace with your YOLOv11 weights

# --- PROCESS IMAGES ---
image_files = sorted(
    [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
)[:1]

for img_path in image_files:
    print(f"Processing {img_path}")
    results = model(img_path)

    # Get annotated image as numpy array
    annotated_img = results[0].plot()  # returns HWC uint8 image

    # Save annotated image manually
    out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
    cv2.imwrite(out_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image: {out_path}")

    # Optional: print summary
    boxes = results[0].boxes
    if len(boxes) > 0:
        print(f"Detections: {len(boxes)} objects")
        for b, c in zip(boxes.xyxy, boxes.cls):
            print(f" - class {int(c)} bbox {b.numpy()}")
    else:
        print("No objects detected")