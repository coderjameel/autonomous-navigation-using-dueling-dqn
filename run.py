import torch
import cv2
import os
import numpy as np
import random
# We import the model class from your training file to ensure it matches exactly
try:
    from train import DuelingDQN 
except ImportError:
    # If your file is named 'train_pro.py', uncomment the line below:
    # from train_pro import DuelingDQN
    pass

# --- CONFIG ---
MODEL_PATH = './save_pro/best_model.pth'
DATA_DIR = './data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
ACTIONS = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 0, 255)} 

def predict_and_show():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No model found at {MODEL_PATH}")
        return

    print(f"Loading neural network from {MODEL_PATH}...")
    
    # Initialize model
    # Note: We assume DuelingDQN is defined in 'train.py' or imported above
    try:
        model = DuelingDQN().to(DEVICE)
    except NameError:
        print("Error: Could not find 'DuelingDQN' class.")
        print("Make sure your training file is named 'train.py' and is in this folder.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 
    
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    random.shuffle(all_files)
    
    print(f"Loaded {len(all_files)} images. Controls: 'n' = Next, 'q' = Quit")
    
    for img_name in all_files:
        img_path = os.path.join(DATA_DIR, img_name)
        frame = cv2.imread(img_path)
        if frame is None: continue

        # --- PREPROCESSING (THE FIX IS HERE) ---
        # 1. Resize to 200x66 (Width, Height) -> Result shape (66, 200, 3)
        input_img = cv2.resize(frame, (200, 66))
        
        # 2. Normalize
        input_img = input_img.astype(np.float32) / 255.0
        
        # 3. To Tensor
        # CRITICAL FIX: We do NOT permute/flip here. We send (Batch, Height, Width, Color)
        # The model's forward() function handles the flip to (Batch, Color, Height, Width)
        input_tensor = torch.tensor(input_img).unsqueeze(0).to(DEVICE)

        # --- PREDICTION ---
        with torch.no_grad():
            q_values = model(input_tensor)
            prediction = torch.argmax(q_values).item()
            confidence = torch.max(torch.softmax(q_values, dim=1)).item()

        # --- VISUALIZATION ---
        action_text = ACTIONS[prediction]
        color = COLORS[prediction]
        
        # Text
        cv2.putText(frame, f"AI: {action_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Conf: {int(confidence*100)}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Arrow
        h, w = frame.shape[:2]
        cx, cy = w // 2, h - 50
        
        if prediction == 0: # LEFT
            cv2.arrowedLine(frame, (cx, cy), (cx-80, cy), color, 4)
        elif prediction == 2: # RIGHT
            cv2.arrowedLine(frame, (cx, cy), (cx+80, cy), color, 4)
        else: # STRAIGHT
            cv2.arrowedLine(frame, (cx, cy), (cx, cy-80), color, 4)

        cv2.imshow("Deep RL Driver View", frame)
        
        key = cv2.waitKey(0) 
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_and_show()