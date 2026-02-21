import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
DATA_DIR = "./data_precomputed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./training_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- DATASET ---
class DrivingDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".pt")]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.root_dir, self.files[idx]))
        return data["tensor"], torch.tensor(data["angle"], dtype=torch.float32)

def discretize_actions(angles):
    actions = []
    for angle in angles:
        if angle < -0.61:
            actions.append(0)
        elif angle > 3.83:
            actions.append(2)
        else:
            actions.append(1)
    return torch.LongTensor(actions).to(DEVICE)

# --- MODEL ---
class AdvancedDQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 5, 66, 200)
            flat_size = self.conv(dummy).view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ELU(), nn.Linear(512, 1)
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ELU(), nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.adv_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# --- TRAIN ---
def train():
    dataset = DrivingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = AdvancedDQN().to(DEVICE)
    model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    # Metrics storage
    epoch_losses = []
    epoch_accuracies = []
    lr_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, angles in dataloader:
            inputs = inputs.to(DEVICE)
            targets = discretize_actions(angles)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        acc = correct / total

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(acc)
        lr_history.append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "fast_dqn_model.pth"))

    # --- PLOTS ---
    plot_training_curves(epoch_losses, epoch_accuracies, lr_history)

    print("Training complete.")

def plot_training_curves(losses, accuracies, lr_history):
    epochs = range(1, len(losses)+1)

    # Loss
    plt.figure()
    plt.plot(epochs, losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, accuracies)
    plt.title("Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_curve.png"))
    plt.close()

    # Learning Rate
    plt.figure()
    plt.plot(epochs, lr_history)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.savefig(os.path.join(SAVE_DIR, "lr_curve.png"))
    plt.close()

if __name__ == "__main__":
    train()