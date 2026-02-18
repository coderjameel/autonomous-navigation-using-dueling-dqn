import os
import matplotlib.pyplot as plt

DATA_DIR = './data'
angles = []

with open(os.path.join('data.txt')) as f:
    for line in f:
        try:
            # format: 0.jpg 0.25,2018...
            angle = float(line.split()[1].split(',')[0])
            angles.append(angle)
        except: continue

# Count them based on your current threshold
left = len([a for a in angles if a < -0.15])
right = len([a for a in angles if a > 0.15])
straight = len([a for a in angles if -0.15 <= a <= 0.15])

print(f"TOTAL SAMPLES: {len(angles)}")
print(f"LEFT:     {left} ({left/len(angles)*100:.1f}%)")
print(f"RIGHT:    {right} ({right/len(angles)*100:.1f}%)")
print(f"STRAIGHT: {straight} ({straight/len(angles)*100:.1f}%)")

# If Straight is < 10%, your model will ignore it.