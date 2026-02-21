import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Increase to 64 if GPU memory allows

DATA_DIR = "./data"
SAVE_DIR = "./data_precomputed"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading models...")
seg_model = YOLO("yolov8n-seg.pt")
depth_estimator = pipeline(
    task="depth-estimation",
    model="Intel/dpt-hybrid-midas",
    device=0 if DEVICE == "cuda" else -1
)

# Load metadata
samples = []
with open(os.path.join(DATA_DIR, "data.txt"), "r") as f:
    for line in f:
        parts = line.strip().split()
        img_name = parts[0]
        angle = float(parts[1].split(",")[0])
        samples.append((img_name, angle))

def process_batch(batch):
    images = []
    valid_entries = []

    for img_name, angle in batch:
        img_path = os.path.join(DATA_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (200, 66))
        images.append(image)
        valid_entries.append((img_name, angle))

    if len(images) == 0:
        return

    # --- YOLO Batch ---
    yolo_results = seg_model(images, verbose=False)

    # --- Depth Batch ---
    pil_batch = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    depth_outputs = depth_estimator(pil_batch)

    for i, (img_name, angle) in enumerate(valid_entries):
        image = images[i]

        # Segmentation
        if yolo_results[i].masks is not None:
            seg_mask = torch.sum(yolo_results[i].masks.data, dim=0)
            seg_mask = (seg_mask > 0).float().cpu().numpy()
            seg_mask = cv2.resize(seg_mask, (200, 66))
        else:
            seg_mask = np.zeros((66, 200), dtype=np.float32)

        # Depth
        depth_map = np.array(depth_outputs[i]["depth"])
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = cv2.resize(depth_map, (200, 66))

        rgb = image.astype(np.float32) / 255.0
        combined = np.dstack((rgb, seg_mask, depth_map))

        torch.save({
            "tensor": torch.from_numpy(combined).permute(2, 0, 1).float(),
            "angle": angle
        }, os.path.join(SAVE_DIR, img_name.replace(".jpg", ".pt")))

print("Starting batched preprocessing...")

for i in tqdm(range(0, len(samples), BATCH_SIZE)):
    process_batch(samples[i:i+BATCH_SIZE])

print("Precomputation complete.")