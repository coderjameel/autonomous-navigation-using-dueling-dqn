import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import time
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# --- CONFIGURATION ---
BATCH_SIZE = 64            # Adjusted for stability
EPOCHS = 30
LEARNING_RATE = 1e-4
LOG_DIR = './save_pro'
DATA_DIR = './data'        # Folder containing images and data.txt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

console = Console()

# --- PART 1: CUSTOM DATA LOADER (Reads your specific text format) ---
class DrivingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        txt_path = 'data.txt'
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Cannot find data.txt in {root_dir}")

        with open(txt_path, 'r') as f:
            for line in f:
                # Format: 0.jpg 0.000000,2018-07-01 17:09:44:912
                try:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        # The angle is the part before the comma in the second block
                        angle = float(parts[1].split(',')[0]) 
                        self.samples.append((img_name, angle))
                except Exception as e:
                    continue # Skip bad lines

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, angle = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load Image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if file is missing (prevents crash)
            image = np.zeros((66, 200, 3), dtype=np.uint8)
        
        # Resize to Nvidia Standard (66x200) for Dueling DQN
        image = cv2.resize(image, (200, 66)) 
        
        # Convert to Float & Normalize (0-1)
        image = image.astype(np.float32) / 255.0
        
        # Return: Image (Tensor), Angle (Float)
        return torch.tensor(image), torch.tensor(angle, dtype=torch.float32)

# --- PART 2: THE DUELING DQN MODEL ---
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Feature Extractor
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        
        # Dummy pass to calculate flat size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            self.flatten_size = x.view(1, -1).size(1)

        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage Stream (Left, Straight, Right)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3) 
        )

    def forward(self, x):
        # Permute: (Batch, H, W, C) -> (Batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        
        return val + (adv - adv.mean(dim=1, keepdim=True))

# --- HELPER: CONVERT ANGLES TO ACTIONS ---
# --- HELPER: CONVERT ANGLES TO ACTIONS ---
def discretize_actions(angles):
    """
    BALANCED DATASET LOGIC:
    0 = Left     (Angle < -0.61)
    1 = Straight (Angle between -0.61 and 3.83)
    2 = Right    (Angle > 3.83)
    """
    actions = []
    for angle in angles:
        if angle < -0.61: 
            actions.append(0) # Left
        elif angle > 3.83: 
            actions.append(2) # Right
        else:
            actions.append(1) # Straight (Middle 33% of data)
    return torch.LongTensor(actions).to(DEVICE)

# --- VISUALIZATION HELPERS ---
def make_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(Layout(name="stats"), Layout(name="logs"))
    return layout

def create_stats_table(epoch, step, loss, best_loss):
    table = Table(title="Training Metrics", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")
    table.add_row("Current Epoch", f"{epoch}/{EPOCHS}")
    table.add_row("Global Step", str(step))
    table.add_row("Current Loss", f"{loss:.6f}")
    table.add_row("Best Loss", f"{best_loss:.6f}")
    table.add_row("Device", str(DEVICE).upper())
    return Panel(table, title="SYSTEM STATUS", border_style="blue")

def create_log_panel(log_list):
    return Panel("\n".join(log_list[-15:]), title="EXECUTION LOGS", border_style="yellow")

# --- MAIN TRAINING LOOP ---
def train():
    # 1. SETUP DATA
    if not os.path.exists(DATA_DIR):
        console.print(f"[bold red]ERROR: Data directory '{DATA_DIR}' not found![/bold red]")
        return

    dataset = DrivingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    console.print(f"[green]Loaded {len(dataset)} images from {DATA_DIR}[/green]")

    # 2. SETUP MODEL
    model = DuelingDQN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss() # Changed to CE for classification

    layout = make_layout()
    layout["header"].update(Panel("🚀 DEEP RL DRIVER - DUELING DQN", style="bold white on blue"))
    
    logs = [f"[green]Starting training on {DEVICE}...[/green]"]
    best_loss = float('inf')
    global_step = 0
    
    progress = Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    task_id = progress.add_task("[cyan]Training...", total=EPOCHS)

    with Live(layout, refresh_per_second=4) as live:
        for epoch in range(EPOCHS):
            epoch_loss = 0
            
            for i, (images, angles) in enumerate(dataloader):
                start_time = time.time()
                
                # Move to GPU
                images = images.to(DEVICE)
                
                # Create Ground Truth (Expert Actions)
                # We turn the continuous angle into Class 0, 1, or 2
                expert_actions = discretize_actions(angles)
                
                # Forward Pass
                q_values = model(images) # Output: [Batch, 3]
                
                # Loss Calculation
                # We want the Q-values to match the Expert's Class
                # Using CrossEntropy is a stable proxy for "Large Margin Classification" in Offline RL
                loss = loss_fn(q_values, expert_actions)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                global_step += 1
                
                if current_loss < best_loss:
                    best_loss = current_loss

                # Update Logs
                if i % 10 == 0:
                    dt = (time.time() - start_time) * 1000
                    logs.append(f"[dim]Step {global_step}: Loss {current_loss:.4f} | Batch {dt:.1f}ms[/dim]")
                
                layout["main"]["stats"].update(create_stats_table(epoch+1, global_step, current_loss, best_loss))
                layout["main"]["logs"].update(create_log_panel(logs))
            
            progress.advance(task_id)
            
            # Save Checkpoint
            if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pth"))
            logs.append(f"[bold yellow]Epoch {epoch+1} Saved.[/bold yellow]")

    console.print("[bold green]TRAINING COMPLETE.[/bold green]")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        console.print("[bold red]Stopped by User[/bold red]")