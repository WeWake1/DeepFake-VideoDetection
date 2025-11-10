"""
All-in-one training script for DeepFake detection (single-file).

This script contains:
- Fixed dataset with proper flow loading and validation
- Enhanced model with 2-stream processing and dual ConvLSTM
- Improved training with scheduler, longer epochs, and better initialization
- Critical fixes for flow handling and sequence alignment

Run with:
H:\IDPS\Flash-IDS\.venv\Scripts\python.exe train_all_in_one.py --frames-dir "H:\DeepFake\DeepFake-VideoDetection\Data" --flows-dir "H:\DeepFake\DeepFake-VideoDetection\Flows_OpenCV" --use-flow --epochs 50 --batch-size 4

Note: Default settings are tuned for 24GB GPU. If you get OOM, reduce batch size.
"""

from __future__ import annotations
import os
import argparse
import time
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
import imghdr

# ---------------------- Critical Fix: Flow Loading ----------------------
def read_flo(path: str) -> np.ndarray:
    """Read Middlebury .flo file (OpenCV compatible format)"""
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f'Invalid .flo file: {path} (magic={magic})')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        data = np.reshape(data, (h, w, 2))
        return data

def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname)[1].lower()
    return ext in ('.jpg', '.jpeg', '.png', '.bmp')

def is_valid_image(path: str) -> bool:
    """Quick check if PIL can verify the image (doesn't load pixel data)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class AllInOneDataset(Dataset):
    """Scans frames_root for subfolders containing images.
    Each folder is treated as one video and labeled as fake if the path contains 'fake'.
    Returns sequences of length seq_len (T), and optional flow (T-1 pairs) if flows_root provided.
    """
    def __init__(self, frames_root: str, seq_len: int = 16, size: int = 224, flows_root: Optional[str] = None):
        self.frames_root = frames_root
        self.seq_len = seq_len
        self.size = size
        self.flows_root = flows_root
        self.min_frames = 2 if flows_root else 1  # Require 2 frames for flow

        # collect video folders (directories that contain at least min_frames)
        self.video_dirs: List[str] = []
        self.labels: List[int] = []
        for root, _, files in os.walk(frames_root):
            img_files = [f for f in files if is_image_file(f)]
            if len(img_files) >= self.min_frames:
                self.video_dirs.append(root)
                self.labels.append(1 if 'fake' in root.lower() else 0)

        if len(self.video_dirs) == 0:
            raise RuntimeError(f'No valid video folders found under: {frames_root} (min {self.min_frames} frames)')
        
        print(f"Found {len(self.video_dirs)} valid videos (min {self.min_frames} frames each)")

        self.transform = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_dirs)

    def _get_frame_paths(self, video_dir: str) -> List[str]:
        files = [f for f in os.listdir(video_dir) if is_image_file(f)]
        files.sort()
        return [os.path.join(video_dir, f) for f in files]

    def _flow_for_pair(self, frame_a: str, frame_b: str) -> Optional[np.ndarray]:
        if self.flows_root is None:
            return None
            
        rel = os.path.relpath(os.path.dirname(frame_a), self.frames_root)
        flow_dir = os.path.join(self.flows_root, rel)
        
        if not os.path.exists(flow_dir):
            return None
            
        # Handle both naming conventions
        base_a = os.path.splitext(os.path.basename(frame_a))[0]
        base_b = os.path.splitext(os.path.basename(frame_b))[0]
        flow_path1 = os.path.join(flow_dir, f"{base_a}_{base_b}.flo")
        flow_path2 = os.path.join(flow_dir, f"{base_a}__{base_b}.flo")
        
        if os.path.exists(flow_path1):
            return read_flo(flow_path1)
        elif os.path.exists(flow_path2):
            return read_flo(flow_path2)
        return None

    def __getitem__(self, idx: int):
        video_dir = self.video_dirs[idx]
        label = self.labels[idx]
        frames = self._get_frame_paths(video_dir)
        
        # Ensure we have enough frames
        n = len(frames)
        if n < self.min_frames:
            # Shouldn't happen due to filtering, but safety check
            frames = frames * (self.min_frames // n + 1)
            n = len(frames)

        # Sample indices evenly
        indices = np.linspace(0, n - 1, num=self.seq_len).astype(int)
        
        # Load frames
        imgs = []
        valid_frames = []
        for i in indices:
            p = frames[i]
            try:
                img = Image.open(p).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                valid_frames.append(p)
            except Exception as e:
                print(f"Warning: Failed to load {p} ({e}), using zero tensor")
                imgs.append(torch.zeros(3, self.size, self.size))
                valid_frames.append(None)

        seq = torch.stack(imgs, dim=0)  # (T, C, H, W)
        
        # Load flows
        flow_seq = []
        for i in range(len(valid_frames) - 1):
            if valid_frames[i] is None or valid_frames[i+1] is None:
                flow_seq.append(torch.zeros(2, self.size, self.size))
                continue
                
            f = self._flow_for_pair(valid_frames[i], valid_frames[i+1])
            if f is None:
                flow_seq.append(torch.zeros(2, self.size, self.size))
            else:
                # Resize flow to match input size
                fh, fw = f.shape[:2]
                u = Image.fromarray(f[:, :, 0]).resize((self.size, self.size), resample=Image.BILINEAR)
                v = Image.fromarray(f[:, :, 1]).resize((self.size, self.size), resample=Image.BILINEAR)
                fu = np.array(u, dtype=np.float32)
                fv = np.array(v, dtype=np.float32)
                flow_arr = np.stack([fu, fv], axis=0)  # (2,H,W)
                flow_seq.append(torch.from_numpy(flow_arr))
        
        # Convert to tensor and pad if needed
        if len(flow_seq) < self.seq_len - 1:
            padding = [torch.zeros(2, self.size, self.size)] * (self.seq_len - 1 - len(flow_seq))
            flow_seq.extend(padding)
        flow_tensor = torch.stack(flow_seq, dim=0)  # (T-1, 2, H, W)
        
        return seq, flow_tensor, torch.tensor(label, dtype=torch.float32), video_dir

# ---------------------- Enhanced Model Architecture ----------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, 
                             kernel_size, padding=padding, bias=False)
        self.hidden_channels = hidden_channels

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class TemporalAttention(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.w = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, seq_feats):
        scores = self.w(seq_feats)
        weights = torch.softmax(scores, dim=1)
        context = (weights * seq_feats).sum(dim=1)
        return context, weights

class TwoStreamDetector(nn.Module):
    def __init__(self, use_flow: bool = False, feat_channels: int = 512, hidden_channels: int = 512):
        super().__init__()
        self.use_flow = use_flow
        self.hidden_channels = hidden_channels
        
        # RGB backbone
        backbone = models.resnet50(pretrained=True)
        self.rgb_backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.rgb_reduce = nn.Conv2d(2048, feat_channels, kernel_size=1, bias=False)
        self.rgb_bn = nn.BatchNorm2d(feat_channels)
        
        # Flow backbone (deeper architecture)
        if use_flow:
            self.flow_encoder = nn.Sequential(
                nn.Conv2d(2, 64, 5, padding=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 5, padding=2, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 5, padding=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, feat_channels, 5, padding=2, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True)
            )
        
        # Dual ConvLSTM layers for better temporal modeling
        self.conv_lstm1 = ConvLSTMCell(feat_channels, hidden_channels)
        self.conv_lstm2 = ConvLSTMCell(hidden_channels, hidden_channels)
        
        self.attn = TemporalAttention(hidden_channels)
        
        if use_flow:
            self.flow_conv_lstm1 = ConvLSTMCell(feat_channels, hidden_channels)
            self.flow_conv_lstm2 = ConvLSTMCell(hidden_channels, hidden_channels)
            self.flow_attn = TemporalAttention(hidden_channels)
            final_dim = hidden_channels * 2
        else:
            final_dim = hidden_channels

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, seq_rgb, seq_flow=None):
        # seq_rgb: (B, T, 3, H, W)
        B, T, C, H, W = seq_rgb.shape
        device = seq_rgb.device
        
        # RGB stream
        h1, c1 = None, None
        h2, c2 = None, None
        rgb_pooled = []
        
        for t in range(T):
            frame = seq_rgb[:, t]
            feat = self.rgb_backbone(frame)
            feat = self.rgb_bn(self.rgb_reduce(feat))
            
            # Initialize hidden states
            if h1 is None:
                h1 = torch.zeros(B, self.hidden_channels, feat.size(2), feat.size(3), device=device)
                c1 = torch.zeros_like(h1)
                h2 = torch.zeros(B, self.hidden_channels, feat.size(2), feat.size(3), device=device)
                c2 = torch.zeros_like(h2)
            
            # Dual-layer ConvLSTM
            h1, c1 = self.conv_lstm1(feat, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
            pooled = h2.mean(dim=[2, 3])
            rgb_pooled.append(pooled)
        
        rgb_seq_feats = torch.stack(rgb_pooled, dim=1)
        rgb_context, rgb_weights = self.attn(rgb_seq_feats)

        # Flow stream
        if self.use_flow and seq_flow is not None and seq_flow.numel() > 0:
            steps = seq_flow.shape[1]  # T-1
            h1_f, c1_f = None, None
            h2_f, c2_f = None, None
            flow_pooled = []
            
            for t in range(steps):
                f = seq_flow[:, t].to(device)
                feat_f = self.flow_encoder(f)
                
                # Initialize hidden states
                if h1_f is None:
                    h1_f = torch.zeros(B, self.hidden_channels, feat_f.size(2), feat_f.size(3), device=device)
                    c1_f = torch.zeros_like(h1_f)
                    h2_f = torch.zeros(B, self.hidden_channels, feat_f.size(2), feat_f.size(3), device=device)
                    c2_f = torch.zeros_like(h2_f)
                
                # Dual-layer ConvLSTM for flow
                h1_f, c1_f = self.flow_conv_lstm1(feat_f, h1_f, c1_f)
                h2_f, c2_f = self.flow_conv_lstm2(h1_f, h2_f, c2_f)
                pooled_f = h2_f.mean(dim=[2, 3])
                flow_pooled.append(pooled_f)
            
            flow_seq_feats = torch.stack(flow_pooled, dim=1)
            flow_context, flow_weights = self.flow_attn(flow_seq_feats)
            context = torch.cat([rgb_context, flow_context], dim=1)
        else:
            context = rgb_context

        logits = self.classifier(context).squeeze(1)
        return logits

# ---------------------- Improved Training ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-dir', default='Data', required=False, help='Path to frames root (default=Data)')
    p.add_argument('--flows-dir', default=None)
    p.add_argument('--use-flow', action='store_true')
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--save-dir', default='checkpoints')
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--verbose', type=int, default=1, help='Verbosity level: 0=min, 1=epoch summary, 2=full details')
    return p.parse_args()


def check_cuda():
    print('torch', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print('total VRAM (GB):', props.total_memory / 1024**3)


def train():
    args = parse_args()
    check_cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
 
    flows_root = args.flows_dir if args.use_flow else None
    dataset = AllInOneDataset(args.frames_dir, seq_len=args.seq_len, size=args.size, flows_root=flows_root)
    
    # Create train/val split with shuffling
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    
    n_val = max(1, int(n * args.val_split))
    train_idx = idxs[:-n_val]
    val_idx = idxs[-n_val:]
    
    print(f"Training on {len(train_idx)} videos, validating on {len(val_idx)} videos")
    
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=max(0, args.num_workers//2), pin_memory=True)

    model = TwoStreamDetector(use_flow=(flows_root is not None))
    model = model.to(device)
    
    # Improved optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))  # Handle class imbalance
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        
        for batch in loader:
            seqs, flows, labels, paths = batch
            seqs = seqs.to(device)
            labels = labels.to(device)
            flows = flows.to(device) if args.use_flow else None

            optimizer.zero_grad()
            with autocast():
                logits = model(seqs, flows)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * seqs.size(0)
        
        train_loss = total_loss / len(train_ds)
        elapsed = time.time() - start
        # basic epoch summary
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} time={elapsed:.1f}s')
 
        # Validation
        model.eval()
        val_loss = 0.0
        first_batch_info = None
        with torch.no_grad():
            for bidx, batch in enumerate(val_loader):
                seqs, flows, labels, paths = batch
                seqs = seqs.to(device)
                labels = labels.to(device)
                flows = flows.to(device) if args.use_flow else None

                with autocast():
                    logits = model(seqs, flows)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * seqs.size(0)

                # capture first batch predictions for verbose output
                if args.verbose >= 2 and first_batch_info is None:
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    first_batch_info = {
                        'paths': paths,
                        'labels': labels.detach().cpu().numpy().tolist(),
                        'probs': probs.tolist()
                    }
        
        val_loss = val_loss / len(val_ds)
        print(f'-> val_loss={val_loss:.4f}')
        # verbose details
        if args.verbose >= 2:
            lr = optimizer.param_groups[0]['lr']
            avg_batch_time = elapsed / max(1, len(loader))
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3 if torch.cuda.is_available() else 0.0
            print(f'   lr={lr:.3e} avg_batch_time={avg_batch_time:.3f}s gpu_alloc={mem_alloc:.3f}GB gpu_reserved={mem_reserved:.3f}GB')
            if first_batch_info is not None:
                print('   Sample validation predictions (first batch):')
                for pth, lab, pr in zip(first_batch_info['paths'][:5], first_batch_info['labels'][:5], first_batch_info['probs'][:5]):
                    print(f'     {pth}  gt={int(lab)}  pred={pr:.4f}')
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.save_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt)
            print(f'Saved best checkpoint: {ckpt} (val_loss={val_loss:.4f})')
    
    print('Training completed. Best validation loss:', best_val)

if __name__ == '__main__':
    train()