import os
import numpy as np

DATA_DIR = './data'
angles = []

print("Reading data...")
with open(os.path.join(DATA_DIR, 'data.txt'), 'r') as f:
    for line in f:
        try:
            # format: 0.jpg 0.25,2018...
            angle = float(line.split()[1].split(',')[0])
            angles.append(angle)
        except: continue

# Sort all angles from negative (Left) to positive (Right)
angles.sort()
total = len(angles)

# We want 3 equal chunks. 
# The first cut is at 33% (1/3rd). The second cut is at 66% (2/3rds).
index_33 = int(total * 0.33)
index_66 = int(total * 0.66)

threshold_left = angles[index_33]
threshold_right = angles[index_66]

print(f"--- BALANCED THRESHOLDS FOUND ---")
print(f"To get ~33% in each class, use these values:")
print(f"LEFT LIMIT:  {threshold_left:.4f}")
print(f"RIGHT LIMIT: {threshold_right:.4f}")
print("-" * 30)
print(f"Logic for train_pro.py:")
print(f"if angle < {threshold_left:.4f}: return 0 (LEFT)")
print(f"elif angle > {threshold_right:.4f}: return 2 (RIGHT)")
print(f"else: return 1 (STRAIGHT)")