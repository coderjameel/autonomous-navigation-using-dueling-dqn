import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from transformers import DPTImageProcessor, DPTForDepthEstimation

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # H200 can handle this easily
DATA_DIR = "./data"
SAVE_DIR = "./data_precomputed"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# ---------------- LOAD MODELS ----------------
print("Loading YOLO...")
seg_model = YOLO("yolov8n-seg.pt")

print("Loading Depth Model (Direct, no pipeline)...")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(DEVICE)
depth_model.eval()

# ---------------- LOAD DATA ----------------
samples = []
with open(os.path.join(DATA_DIR, "data.txt"), "r") as f:
    for line in f:
        parts = line.strip().split()
        img_name = parts[0]
        angle = float(parts[1].split(",")[0])
        samples.append((img_name, angle))

print(f"Total images: {len(samples)}")

# ---------------- PROCESS BATCH ----------------
def process_batch(batch):
    images = []
    angles = []
    names = []

    for img_name, angle in batch:
        img_path = os.path.join(DATA_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (200, 66))
        images.append(img)
        angles.append(angle)
        names.append(img_name)

    if len(images) == 0:
        return

    # ---------- YOLO BATCH ----------
    yolo_results = seg_model(images, verbose=False)

    # ---------- DEPTH BATCH (TRUE GPU BATCH) ----------
    pil_batch = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    inputs = processor(images=pil_batch, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(66, 200),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)

    depth_maps = predicted_depth.cpu().numpy()

    # ---------- SAVE ----------
    for i in range(len(images)):
        img = images[i]

        # Segmentation
        if yolo_results[i].masks is not None:
            seg_mask = torch.sum(yolo_results[i].masks.data, dim=0)
            seg_mask = (seg_mask > 0).float().cpu().numpy()
            seg_mask = cv2.resize(seg_mask, (200, 66))
        else:
            seg_mask = np.zeros((66, 200), dtype=np.float32)

        depth_map = depth_maps[i]
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

        rgb = img.astype(np.float32) / 255.0
        combined = np.dstack((rgb, seg_mask, depth_map))

        torch.save({
            "tensor": torch.from_numpy(combined).permute(2, 0, 1).float(),
            "angle": angles[i]
        }, os.path.join(SAVE_DIR, names[i].replace(".jpg", ".pt")))

# ---------------- MAIN LOOP ----------------
print("Starting FAST batched preprocessing...")

for i in tqdm(range(0, len(samples), BATCH_SIZE)):
    process_batch(samples[i:i+BATCH_SIZE])

print("Precomputation complete.")