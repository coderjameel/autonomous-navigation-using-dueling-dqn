import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

txt_path = os.path.join(DATA_DIR, "data.txt")

samples = []
with open(txt_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        img_name = parts[0]
        angle = float(parts[1].split(",")[0])
        samples.append((img_name, angle))

for img_name, angle in tqdm(samples):
    img_path = os.path.join(DATA_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    image = cv2.resize(image, (200, 66))

    # --- SEGMENTATION ---
    results = seg_model(image, verbose=False)
    if results[0].masks is not None:
        seg_mask = torch.sum(results[0].masks.data, dim=0)
        seg_mask = (seg_mask > 0).float().cpu().numpy()
        seg_mask = cv2.resize(seg_mask, (200, 66))
    else:
        seg_mask = np.zeros((66, 200), dtype=np.float32)

    # --- DEPTH ---
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = depth_estimator(Image.fromarray(rgb))
    depth_map = np.array(depth["depth"])
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    depth_map = cv2.resize(depth_map, (200, 66))

    rgb = image.astype(np.float32) / 255.0
    combined = np.dstack((rgb, seg_mask, depth_map))

    torch.save({
        "tensor": torch.from_numpy(combined).permute(2, 0, 1).float(),
        "angle": angle
    }, os.path.join(SAVE_DIR, img_name.replace(".jpg", ".pt")))

print("Precomputation complete.")