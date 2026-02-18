import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
LOG_DIR = './results'         # Where to save graphs/models
DATA_DIR = './data'           # Where images are
CHECKPOINT_PATH = './results/checkpoint.pth' # File to save/resume progress
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
console = Console()

# --- 1. DATASET & HELPERS ---
class DrivingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.angles = [] 
        
        txt_path = os.path.join(root_dir, 'data.txt')
        if not os.path.exists(txt_path):
             # Create dummy if missing to prevent crash during copy-paste
             console.print(f"[red]Warning: {txt_path} not found![/red]")
             return

        with open(txt_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    img_name = parts[0]
                    angle = float(parts[1].split(',')[0])
                    self.samples.append((img_name, angle))
                    self.angles.append(angle)
                except: continue

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_name, angle = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: image = np.zeros((66, 200, 3), dtype=np.uint8)
        
        image = cv2.resize(image, (200, 66))
        image = image.astype(np.float32) / 255.0
        return torch.tensor(image), torch.tensor(angle, dtype=torch.float32)

def discretize_actions(angles):
    # CALIBRATED THRESHOLDS from your data analysis
    actions = []
    for angle in angles:
        if angle < -0.61: actions.append(0)   # Left
        elif angle > 3.83: actions.append(2)  # Right
        else: actions.append(1)               # Straight
    return torch.LongTensor(actions).to(DEVICE)

# --- 2. MODEL ARCHITECTURE ---
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            self.flat_size = x.view(1, -1).size(1)

        self.value_stream = nn.Sequential(nn.Linear(self.flat_size, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(self.flat_size, 512), nn.ReLU(), nn.Linear(512, 3))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# --- 3. RICH UI LAYOUT ---
def make_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=5)
    )
    layout["main"].split_row(
        Layout(name="stats"),
        Layout(name="logs")
    )
    return layout

def create_stats_table(epoch, step, loss, best_loss, lr):
    table = Table(title="Training Metrics", style="cyan", expand=True)
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")
    table.add_row("Epoch", f"{epoch}/{EPOCHS}")
    table.add_row("Global Step", str(step))
    table.add_row("Current Loss", f"{loss:.4f}")
    table.add_row("Best Loss", f"{best_loss:.4f}")
    table.add_row("LR", f"{lr:.2e}")
    return Panel(table, title="SYSTEM STATUS", border_style="blue")

def generate_report(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss', color='#FF5733')
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join(LOG_DIR, 'loss_curve.png'))
    plt.close()

# --- 4. MAIN TRAINING LOOP ---
def train():
    # Setup
    dataset = DrivingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = DuelingDQN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # State Variables
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    loss_history = []
    logs = []

    # --- RESUME LOGIC ---
    if os.path.exists(CHECKPOINT_PATH):
        console.print(f"[bold yellow]Found checkpoint at {CHECKPOINT_PATH}. Resuming...[/bold yellow]")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        loss_history = checkpoint.get('loss_history', [])
        logs.append(f"[yellow]Resumed from Epoch {start_epoch}[/yellow]")
    else:
        logs.append("[green]Starting Fresh Training...[/green]")

    # Setup Rich UI
    layout = make_layout()
    layout["header"].update(Panel("🚀 AUTONOMOUS DRIVING AI - TRAINING DASHBOARD", style="bold white on blue"))
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•", TransferSpeedColumn(), "•", TimeRemainingColumn(),
        expand=True
    )
    task_id = progress.add_task("[cyan]Training...", total=(EPOCHS - start_epoch) * len(dataloader))
    layout["footer"].update(Panel(progress, border_style="green"))

    # THE LOOP
    with Live(layout, refresh_per_second=4) as live:
        for epoch in range(start_epoch, EPOCHS):
            epoch_loss = 0
            
            for i, (images, angles) in enumerate(dataloader):
                start_time = time.time()
                
                # Train Step
                images = images.to(DEVICE)
                expert_actions = discretize_actions(angles)
                
                optimizer.zero_grad()
                q_values = model(images)
                loss = loss_fn(q_values, expert_actions)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                loss_history.append(current_loss)
                global_step += 1
                
                # Update Best Model
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pth"))

                # Update Logs (Throttle to avoid flicker)
                if i % 10 == 0:
                    dt = (time.time() - start_time) * 1000
                    logs.append(f"[dim]Ep {epoch} Step {i}: Loss {current_loss:.4f} ({dt:.0f}ms)[/dim]")
                    if len(logs) > 15: logs.pop(0)

                # Update UI
                layout["main"]["stats"].update(create_stats_table(epoch, global_step, current_loss, best_loss, LEARNING_RATE))
                layout["main"]["logs"].update(Panel("\n".join(logs), title="LIVE LOGS", border_style="yellow"))
                progress.advance(task_id)

            # SAVE CHECKPOINT (End of Epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'loss_history': loss_history
            }, CHECKPOINT_PATH)
            
            logs.append(f"[bold green]Epoch {epoch} Complete. Checkpoint Saved.[/bold green]")

    # Finish
    generate_report(loss_history)
    console.print(f"[bold green]TRAINING COMPLETE. Graphs saved to {LOG_DIR}[/bold green]")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        console.print("[bold red]Stopped by user. Progress saved in checkpoint.[/bold red]")