import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import warnings
import shutil
from PIL import Image  # <--- FIXED: Added PIL Import
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    TimeRemainingColumn, TransferSpeedColumn
)

# --- CONFIGURATION ---
BATCH_SIZE = 32         
EPOCHS = 50
LEARNING_RATE = 1e-4
LOG_DIR = './research_results'
DATA_DIR = './data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
CHECKPOINT_PATH = os.path.join(LOG_DIR, 'last_checkpoint.pth')
BEST_MODEL_PATH = os.path.join(LOG_DIR, 'best_model.pth')
CSV_LOG_PATH = os.path.join(LOG_DIR, 'training_log.csv')

# Matplotlib headless mode
plt.switch_backend('agg') 
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
console = Console()
warnings.filterwarnings("ignore")

# --- 1. PERCEPTION ENGINE (YOLO + DEPTH) ---
class PerceptionEngine:
    def __init__(self):
        console.print(f"[bold yellow]Initializing Perception Engine on {DEVICE}...[/bold yellow]")
        
        # 1. YOLOv8 Nano for Segmentation
        from ultralytics import YOLO
        self.seg_model = YOLO("yolov8n-seg.pt")
        
        # 2. MiDaS Small for Depth
        from transformers import pipeline
        # device=0 for CUDA, -1 for CPU
        hf_device = 0 if torch.cuda.is_available() else -1
        self.depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas", device=hf_device)
        
        console.print("[bold green]Perception Engine Online.[/bold green]")

    def process(self, frame):
        """
        Input: numpy array (H, W, 3) BGR
        Output: torch tensor (5, H, W) -> [R, G, B, Seg, Depth]
        """
        # Resize first to save compute
        frame_resized = cv2.resize(frame, (200, 66))
        
        # A. SEGMENTATION (YOLO)
        # verbose=False prevents spamming the console
        results = self.seg_model(frame_resized, verbose=False)
        
        if results[0].masks is not None:
            # Flatten all masks into one binary map
            seg_mask = torch.sum(results[0].masks.data, dim=0)
            seg_mask = (seg_mask > 0).float().cpu().numpy()
            # YOLO output size might differ slightly, force resize back to 66x200
            seg_mask = cv2.resize(seg_mask, (200, 66))
        else:
            seg_mask = np.zeros((66, 200), dtype=np.float32)

        # B. DEPTH ESTIMATION (FIXED HERE)
        # Convert BGR (OpenCV) to RGB
        rgb_numpy = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # FIXED: Convert NumPy Array -> PIL Image
        pil_img = Image.fromarray(rgb_numpy)
        
        # Now pass the PIL Image to the transformer
        depth_out = self.depth_estimator(pil_img) 
        depth_map = np.array(depth_out["depth"])
        
        # Normalize Depth (0-1)
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        depth_map = cv2.resize(depth_map, (200, 66))

        # C. COMPILE CHANNELS
        rgb = frame_resized.astype(np.float32) / 255.0
        
        # Stack: (66, 200, 3) + (66, 200) + (66, 200) -> (66, 200, 5)
        combined = np.dstack((rgb, seg_mask, depth_map))
        
        # To Tensor: (5, 66, 200)
        return torch.from_numpy(combined).permute(2, 0, 1).float()

# Global instance
perception = None 

# --- 2. DATASET ---
class DrivingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        # Initialize perception engine only once
        global perception
        if perception is None:
            perception = PerceptionEngine()

        txt_path = os.path.join(root_dir, 'data.txt')
        if not os.path.exists(txt_path):
             console.print(f"[bold red]CRITICAL: {txt_path} not found![/bold red]")
             return

        with open(txt_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    img_name = parts[0]
                    angle = float(parts[1].split(',')[0])
                    self.samples.append((img_name, angle))
                except: continue

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_name, angle = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None: image = np.zeros((66, 200, 3), dtype=np.uint8)
        
        # --- HERE IS WHERE YOLO IS USED ---
        tensor_data = perception.process(image)
        
        return tensor_data, torch.tensor(angle, dtype=torch.float32)

def discretize_actions(angles):
    # 0: Left, 1: Straight, 2: Right
    actions = []
    for angle in angles:
        if angle < -0.61: actions.append(0)
        elif angle > 3.83: actions.append(2)
        else: actions.append(1)
    return torch.LongTensor(actions).to(DEVICE)

# --- 3. MODEL (5-CHANNEL INPUT) ---
class AdvancedDQN(nn.Module):
    def __init__(self):
        super(AdvancedDQN, self).__init__()
        
        # Input: 5 Channels (R, G, B, Segmentation, Depth)
        self.conv1 = nn.Conv2d(5, 24, 5, stride=2) 
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 5, 66, 200)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            self.flat_size = x.view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            nn.Linear(self.flat_size, 512), nn.ELU(), nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flat_size, 512), nn.ELU(), nn.Linear(512, 3)
        )

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# --- 4. CHECKPOINT MANAGER ---
class CheckpointManager:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def save(self, epoch, global_step, loss, is_best=False):
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        # Save 'latest' for resuming
        torch.save(state, CHECKPOINT_PATH)
        
        # Save 'best' for inference
        if is_best:
            torch.save(state, BEST_MODEL_PATH)
            
    def load(self):
        if not os.path.exists(CHECKPOINT_PATH):
            return 0, 0, float('inf') # start_epoch, start_step, best_loss
        
        console.print(f"[bold cyan]Found checkpoint at {CHECKPOINT_PATH}. Resuming...[/bold cyan]")
        checkpoint = torch.load(CHECKPOINT_PATH)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'] + 1, checkpoint['global_step'], checkpoint.get('loss', float('inf'))

# --- 5. UTILS ---
def init_csv():
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Timestamp', 'Epoch', 'Step', 'Loss', 'Accuracy', 'LR'])

def save_debug_visual(tensor, epoch):
    """Saves visuals of what the agent actually sees (Seg & Depth)"""
    # Tensor: [5, 66, 200]
    seg = tensor[3].cpu().numpy()
    depth = tensor[4].cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(seg, cmap='gray'); plt.title(f'YOLO Mask (Ep {epoch})'); plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap='inferno'); plt.title(f'Depth Map (Ep {epoch})'); plt.axis('off')
    
    debug_dir = os.path.join(LOG_DIR, 'debug_views')
    if not os.path.exists(debug_dir): os.makedirs(debug_dir)
    plt.savefig(os.path.join(debug_dir, f'vision_ep_{epoch}.png'))
    plt.close()

# --- 6. MAIN TRAINING LOOP ---
def make_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=6)
    )
    layout["body"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="logs", ratio=1)
    )
    return layout

def train():
    init_csv()
    dataset = DrivingDataset(DATA_DIR)
    
    if len(dataset) == 0:
        console.print("[red]No data found![/red]")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = AdvancedDQN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialize Checkpoint Manager
    ckpt_manager = CheckpointManager(model, optimizer)
    
    # Resume if possible
    start_epoch, global_step, best_loss = ckpt_manager.load()
    
    # Setup UI
    layout = make_layout()
    layout["header"].update(Panel("[bold white]🏎️  YOLO-DRIVEN AUTONOMOUS AGENT[/bold white]", style="on blue"))
    
    # Progress
    job_progress = Progress(
        "{task.description}", SpinnerColumn(), BarColumn(style="magenta"), 
        "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn()
    )
    epoch_progress = Progress("{task.description}", BarColumn(style="cyan"), TransferSpeedColumn())
    
    total_task = job_progress.add_task("[magenta]Total Training", total=EPOCHS, completed=start_epoch)
    
    progress_group = Table.grid(expand=True)
    progress_group.add_row(Panel(job_progress, border_style="magenta"))
    progress_group.add_row(Panel(epoch_progress, border_style="cyan"))
    layout["footer"].update(progress_group)

    logs_buffer = []

    with Live(layout, refresh_per_second=4) as live:
        for epoch in range(start_epoch, EPOCHS):
            epoch_task = epoch_progress.add_task(f"[cyan]Epoch {epoch+1}", total=len(dataloader))
            
            for i, (inputs, angles) in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                targets = discretize_actions(angles)
                
                optimizer.zero_grad()
                q_values = model(inputs)
                loss = loss_fn(q_values, targets)
                loss.backward()
                optimizer.step()
                
                # Metrics
                preds = torch.argmax(q_values, dim=1)
                acc = (preds == targets).float().mean().item()
                curr_loss = loss.item()
                global_step += 1
                
                # Update UI
                if i % 5 == 0:
                    table = Table(box=None, expand=True)
                    table.add_column("Metric", style="dim"); table.add_column("Value", style="bold")
                    table.add_row("Step", str(global_step))
                    table.add_row("Loss", f"[red]{curr_loss:.4f}[/red]")
                    table.add_row("Best Loss", f"[yellow]{best_loss:.4f}[/yellow]")
                    table.add_row("Accuracy", f"[green]{acc*100:.1f}%[/green]")
                    layout["body"]["metrics"].update(Panel(table, title="LIVE METRICS", border_style="cyan"))
                    
                    logs_buffer.append(f"[dim]Step {global_step}: Loss {curr_loss:.3f}[/dim]")
                    if len(logs_buffer) > 8: logs_buffer.pop(0)
                    layout["body"]["logs"].update(Panel("\n".join(logs_buffer), title="LOGS", border_style="yellow"))

                epoch_progress.advance(epoch_task)

            # End of Epoch: Save Checkpoints
            is_best = curr_loss < best_loss
            if is_best: best_loss = curr_loss
            
            ckpt_manager.save(epoch, global_step, curr_loss, is_best=is_best)
            
            # Save visual debug
            if epoch % 5 == 0: save_debug_visual(inputs[0], epoch)
            
            job_progress.advance(total_task)
            epoch_progress.remove_task(epoch_task)

    console.print(f"[bold green]TRAINING COMPLETE. Best Model: {BEST_MODEL_PATH}[/bold green]")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        console.print("[bold red]Training Paused. Progress Saved.[/bold red]")