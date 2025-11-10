"""
train_defake_production.py

Production-ready training + inference skeleton:
- Generator: UNet per-frame encoder -> ConvLSTM temporal fusion -> UNet-style decoder
- Discriminator: PatchGAN (2D on frames)
- Losses: L1, VGG perceptual, GAN (hinge), Flow consistency (FlowFormer), Detector critic (TwoStreamDetector optional)
- Mixed precision training, checkpointing, validation, inference defake over folders.

Edit CONFIG paths and optionally precompute flows for speed.
"""

import os, time, json, random, math
from pathlib import Path
from glob import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import vgg16
import torchvision.utils as vutils
from PIL import Image

# Optional: LPIPS and face-ID (install separately)
# import lpips
# from facenet_pytorch import InceptionResnetV1 as ArcFaceModel

# Optional: kornia for warping
import kornia

# ---------------- CONFIG ----------------
CONFIG = {
    "ROOT": r"H:\DeepFake\DeepFake-VideoDetection\sofos",                        # project root (change)
    "FAKE_DIR": r"H:\DeepFake\DeepFake-VideoDetection\Data\fake\fake",          # contains subfolders with frames
    "REAL_DIR": r"H:\DeepFake\DeepFake-VideoDetection\Data\real\real",          # contains real subfolders, used if paired
    "MAPPING_JSON": None,                          # path to mapping JSON (real->list(fakes)) OR None
    "SAVE_DIR": r"H:\DeepFake\DeepFake-VideoDetection\sofos\checkpoints_defake",
    "FLOWFORMER_ROOT": r"H:\FlowFormer",           # path to FlowFormer repo (change)
    "FLOW_CHECKPOINT": None,                       # optional FlowFormer ckpt path
    "TWOSTREAM_CHECKPOINT": r"H:\DeepFake\DeepFake-VideoDetection\checkpoints\best_model.pth", # optional critic
    "IMG_SIZE": 256,
    "SEQ_LEN": 5,
    "BATCH_SIZE": 4,
    "EPOCHS": 40,
    "LR": 2e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 4,
    # Loss weights
    "lam_L1": 10.0,
    "lam_perc": 1.0,
    "lam_gan": 1.0,
    "lam_flow": 2.0,
    "lam_det": 1.0,
    # Misc
    "VAL_EVERY_N_EPOCHS": 1,
    "SAVE_EVERY_N_EPOCHS": 1,
    "SEED": 42
}
# ----------------------------------------

# reproducibility
torch.manual_seed(CONFIG["SEED"])
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

# ---------- Helper: FlowFormer loader (adapt to repo) ----------
import sys
FLOWFORMER_AVAILABLE = False
if CONFIG["FLOWFORMER_ROOT"]:
    sys.path.insert(0, CONFIG["FLOWFORMER_ROOT"])
    try:
        # The FlowFormer repo structure varies. Try common import patterns:
        try:
            from core.flowformer import build_flowformer as _build_flowformer
        except Exception:
            from flowformer import build_flowformer as _build_flowformer
        FLOWFORMER_AVAILABLE = True
    except Exception as e:
        print("FlowFormer import failed (you can still precompute flows). Edit FLOWFORMER_ROOT if needed.", e)
        _build_flowformer = None

def build_flowformer(device):
    if not FLOWFORMER_AVAILABLE:
        return None
    model = _build_flowformer(pretrained=False)  # adapt if builder signature is different
    if CONFIG["FLOW_CHECKPOINT"]:
        ck = torch.load(CONFIG["FLOW_CHECKPOINT"], map_location=device)
        if "model" in ck:
            model.load_state_dict(ck["model"])
        else:
            model.load_state_dict(ck)
    model = model.to(device)
    model.eval()
    return model

# ---------- Dataset ----------
class PairedSeqDataset(Dataset):
    """
    If MAPPING_JSON is provided, it should map real_folder -> [fake_folder1, fake_folder2...]
    If not provided, dataset pairs folders by exact name under REAL_DIR and FAKE_DIR.
    Each item returns fake_seq (T,C,H,W) and real_seq (T,C,H,W) if real available, else (fake_seq, fake_seq) for self-reconstruction.
    """
    def __init__(self, fake_root, real_root=None, mapping_json=None, seq_len=5, img_size=256, transform=None):
        self.fake_root = Path(fake_root)
        self.real_root = Path(real_root) if real_root else None
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform
        # Build list of (fake_folder, real_folder_or_None)
        self.items = []
        if mapping_json:
            with open(mapping_json, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for real_key, fake_list in mapping.items():
                real_path = self.real_root / real_key if self.real_root else None
                for fake_name in fake_list:
                    fake_path = self.fake_root / fake_name
                    if fake_path.exists():
                        rp = real_path if real_path and real_path.exists() else None
                        self.items.append((str(fake_path), str(rp) if rp else None))
        else:
            # match by folder name (exact)
            fake_folders = [p for p in self.fake_root.iterdir() if p.is_dir()]
            for f in fake_folders:
                real_candidate = (self.real_root / f.name) if self.real_root else None
                if real_candidate and real_candidate.exists():
                    self.items.append((str(f), str(real_candidate)))
                else:
                    # still include fake-only (unpaired)
                    self.items.append((str(f), None))
        # filter out too-short folders
        self.items = [it for it in self.items if len(list(Path(it[0]).glob("*.[pj][pn]g"))) >= self.seq_len]
        if len(self.items) == 0:
            raise RuntimeError("No dataset items found. Check paths and seq_len.")
    def __len__(self):
        return len(self.items) * 50  # sample multiplier
    def _sorted_frames(self, folder):
        p = Path(folder)
        frames = sorted([x for x in p.iterdir() if x.suffix.lower() in (".png",".jpg",".jpeg")])
        return frames
    def __getitem__(self, idx):
        fake_folder, real_folder = self.items[idx % len(self.items)]
        fake_frames = self._sorted_frames(fake_folder)
        real_frames = self._sorted_frames(real_folder) if real_folder else []
        # choose a subsequence aligned by index (simple)
        max_start = min(len(fake_frames), max(1, len(real_frames) if real_frames else len(fake_frames))) - self.seq_len
        if max_start < 0:
            raise IndexError("Folder too short")
        start = random.randint(0, max_start)
        fake_sel = fake_frames[start:start+self.seq_len]
        # if real exists try to align by same start, else use fake as pseudo-real
        if real_frames and len(real_frames) >= start + self.seq_len:
            real_sel = real_frames[start:start+self.seq_len]
        else:
            real_sel = fake_sel  # fallback (unpaired)
        # load images -> tensors (C,H,W)
        seq_fake = []
        seq_real = []
        for fpath, rpath in zip(fake_sel, real_sel):
            img_f = Image.open(fpath).convert("RGB").resize((self.img_size, self.img_size))
            img_r = Image.open(rpath).convert("RGB").resize((self.img_size, self.img_size))
            tf = T.ToTensor()
            t_f = tf(img_f)
            t_r = tf(img_r)
            if self.transform:
                t_f = self.transform(t_f)
                t_r = self.transform(t_r)
            seq_fake.append(t_f)
            seq_real.append(t_r)
        fake_seq = torch.stack(seq_fake, dim=0)  # (T,C,H,W)
        real_seq = torch.stack(seq_real, dim=0)
        return fake_seq, real_seq

# ---------- Model building blocks ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3

class UNetDecoder(nn.Module):
    def __init__(self, out_ch=3, base=64):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, e1, e2, e3):
        x = self.up3(e3)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        return torch.sigmoid(self.out(x))

# ConvLSTM cell (small, efficient)
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4*hidden_ch, kernel_size, padding=padding)
        self.hidden_ch = hidden_ch
    def forward(self, x, hcur):
        # x: (B,C,H,W), hcur: (h,c)
        h_prev, c_prev = hcur
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_out, self.hidden_ch, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch)
        self.hidden_ch = hidden_ch
    def forward(self, seq):  # seq: (B, T, C, H, W)
        B, T, C, H, W = seq.shape
        device = seq.device
        h = torch.zeros(B, self.hidden_ch, H, W, device=device)
        c = torch.zeros(B, self.hidden_ch, H, W, device=device)
        outs = []
        for t in range(T):
            h, c = self.cell(seq[:,t], (h,c))
            outs.append(h)
        out = torch.stack(outs, dim=1)  # (B,T,hidden,H,W)
        return out

# Full Generator: encode each frame (shared UNetEncoder), temporal ConvLSTM on deepest feature, decode per-frame
class GeneratorUNetConvLSTM(nn.Module):
    def __init__(self, in_ch=3, base=64, hidden_ch=256):
        super().__init__()
        self.enc = UNetEncoder(in_ch, base)
        self.conv_reduce = nn.Conv2d(base*4, hidden_ch, 1)   # reduce channel before ConvLSTM
        self.temporal = ConvLSTM(hidden_ch, hidden_ch//2)    # hidden smaller
        self.conv_expand = nn.Conv2d(hidden_ch//2, base*4, 1)
        self.dec = UNetDecoder(out_ch=3, base=base)
    def forward(self, seq):  # seq: (B,T,C,H,W)
        B,T,C,H,W = seq.shape
        # encode per frame
        e1s, e2s, e3s = [], [], []
        deepest = []
        for t in range(T):
            e1,e2,e3 = self.enc(seq[:,t])
            e1s.append(e1); e2s.append(e2); e3s.append(e3)
            deepest.append(self.conv_reduce(e3))
        deepest = torch.stack(deepest, dim=1)  # (B,T,hidden,H/4,W/4)
        # temporal
        temp_out = self.temporal(deepest)  # (B,T,h2,H/4,W/4)
        # decode each time step from temporal features + skip connections
        out_seq = []
        for t in range(T):
            feat = self.conv_expand(temp_out[:,t])
            # pass through decoder using e1s[t], e2s[t], feat as e3
            out = self.dec(e1s[t], e2s[t], feat)
            out_seq.append(out)
        out_seq = torch.stack(out_seq, dim=1)  # (B,T,3,H,W)
        return out_seq

# PatchGAN Discriminator for frames
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base*2, 4, 2, 1),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2, base*4, 4, 2, 1),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*4, 1, 4, 1, 1)  # no activation, raw score map
        )
    def forward(self, x):
        return self.net(x)  # (B,1,H',W')

# ---------- Loss helpers ----------
class VGGFeat(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.vgg(x)

# Hinge losses
def discriminator_hinge_loss(d_real, d_fake):
    loss_real = torch.mean(F.relu(1. - d_real))
    loss_fake = torch.mean(F.relu(1. + d_fake))
    return loss_real + loss_fake

def generator_hinge_loss(d_fake):
    return -torch.mean(d_fake)

# ---------- Training / Helpers ----------
def validate(G, val_loader, device, vgg, criterion_l1):
    G.eval()
    metrics = {"l1":0.0, "psnr":0.0, "ssim":0.0}
    cnt = 0
    import math
    from skimage.metrics import structural_similarity as ssim
    with torch.no_grad():
        for fake_seq, real_seq in val_loader:
            fake_seq = fake_seq.to(device); real_seq = real_seq.to(device)
            out_seq = G(fake_seq)  # (B,T,3,H,W)
            # center frame metrics
            mid = out_seq.shape[1]//2
            pred = out_seq[:,mid]
            gt = real_seq[:,mid]
            l1v = F.l1_loss(pred, gt).item()
            metrics["l1"] += l1v
            # PSNR/SSIM (on cpu numpy)
            p_np = (pred.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            g_np = (gt.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            # compute average PSNR & SSIM per batch
            for i in range(p_np.shape[0]):
                mse = np.mean((p_np[i].astype(np.float32) - g_np[i].astype(np.float32))**2)
                psnr = 100. if mse==0 else 10*math.log10((255.0**2)/mse)
                metrics["psnr"] += psnr
                try:
                    s = ssim(g_np[i].transpose(1,2,0), p_np[i].transpose(1,2,0), multichannel=True, data_range=255)
                except:
                    s = 0.0
                metrics["ssim"] += s
            cnt += fake_seq.size(0)
            if cnt > 200: break
    # avg
    for k in metrics: metrics[k] /= (cnt if cnt>0 else 1)
    G.train()
    return metrics

def save_checkpoint(state, path):
    torch.save(state, path)

# ---------- Training entrypoint ----------
def train():
    device = torch.device(CONFIG["DEVICE"])
    print("Using device:", device)
    save_dir = Path(CONFIG["SAVE_DIR"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # dataset & loader
    transform = None
    ds = PairedSeqDataset(CONFIG["FAKE_DIR"], CONFIG["REAL_DIR"], mapping_json=CONFIG["MAPPING_JSON"],
                         seq_len=CONFIG["SEQ_LEN"], img_size=CONFIG["IMG_SIZE"], transform=transform)
    # split small val
    n = len(ds)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])

    # models
    G = GeneratorUNetConvLSTM(in_ch=3, base=64, hidden_ch=256).to(device)
    D = PatchDiscriminator().to(device)
    vgg = VGGFeat().to(device)

    # optional FlowFormer model (for flow-consistency). You can also precompute flows and load from disk.
    flow_model = build_flowformer(device) if _build_flowformer is not None else None

    # optional TwoStreamDetector critic (frozen)
    detector = None
    if CONFIG["TWOSTREAM_CHECKPOINT"] and Path(CONFIG["TWOSTREAM_CHECKPOINT"]).exists():
        try:
            # user should implement TwoStreamDetector import and class in another module; here we show placeholder
            from two_stream_detector import TwoStreamDetector  # you have this model in your project
            detector = TwoStreamDetector()
            ck = torch.load(CONFIG["TWOSTREAM_CHECKPOINT"], map_location=device)
            detector.load_state_dict(ck)
            detector = detector.to(device).eval()
            for p in detector.parameters(): p.requires_grad = False
            print("Loaded TwoStreamDetector critic.")
        except Exception as e:
            print("Failed to load TwoStreamDetector; continuing without it.", e)
            detector = None

    # optimizers and scalers
    optG = torch.optim.AdamW(G.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    optD = torch.optim.AdamW(D.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # losses
    l1_loss = nn.L1Loss().to(device)

    best_val = 1e9
    for epoch in range(CONFIG["EPOCHS"]):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        for batch_idx, (fake_seq, real_seq) in enumerate(pbar):
            fake_seq = fake_seq.to(device)  # (B,T,C,H,W)
            real_seq = real_seq.to(device)
            B,T,C,H,W = fake_seq.shape

            # ========== Train D ==========
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    fake_out_seq = G(fake_seq)  # (B,T,3,H,W)
                # sample center frames
                mid = T//2
                real_frame = real_seq[:, mid]
                fake_frame = fake_out_seq[:, mid]
                d_real = D(real_frame)
                d_fake = D(fake_frame.detach())
                lossD = discriminator_hinge_loss(d_real, d_fake)
            optD.zero_grad()
            scaler.scale(lossD).backward()
            scaler.step(optD)
            scaler.update()

            # ========== Train G ==========
            with torch.cuda.amp.autocast():
                fake_out_seq = G(fake_seq)  # (B,T,3,H,W)
                mid = T//2
                fake_frame = fake_out_seq[:, mid]
                real_frame = real_seq[:, mid]
                # GAN loss
                d_fake_for_g = D(fake_frame)
                loss_g_gan = generator_hinge_loss(d_fake_for_g)
                # Recon L1 (sequence)
                loss_l1 = l1_loss(fake_out_seq, real_seq)
                # VGG perceptual (center frame)
                pf = vgg(fake_frame); pr = vgg(real_frame)
                loss_perc = l1_loss(pf, pr)
                # Flow consistency (if flow model available)
                loss_flow = torch.tensor(0.0, device=device)
                if flow_model is not None:
                    # compute flow between consecutive generated frames and between real frames, compare
                    # prepare pairs for flow model: flatten pairs to B*(T-1) compute
                    gen_pairs_a = fake_out_seq[:, :-1].reshape(-1,3,H,W)
                    gen_pairs_b = fake_out_seq[:, 1:].reshape(-1,3,H,W)
                    real_pairs_a = real_seq[:, :-1].reshape(-1,3,H,W)
                    real_pairs_b = real_seq[:, 1:].reshape(-1,3,H,W)
                    try:
                        with torch.no_grad():
                            flow_real = flow_model(real_pairs_a*255.0, real_pairs_b*255.0)  # adapt scale to FlowFormer expectation
                        flow_gen = flow_model(gen_pairs_a*255.0, gen_pairs_b*255.0)
                        # flow outputs shape: (N,2,H,W)
                        loss_flow = F.l1_loss(flow_gen, flow_real)
                    except Exception as e:
                        # if flow_model API fails, set zero and warn
                        print("Flow model error:", e)
                        loss_flow = torch.tensor(0.0, device=device)

                # Detector critic loss (if available)
                loss_det = torch.tensor(0.0, device=device)
                if detector is not None:
                    # detector expects sequence shaped; adapt if needed
                    try:
                        det_pred = detector(fake_out_seq)  # expects logits/probs shape (B,)
                        # we want detector output low (real); if detector returns logits -> sigmoid inside loss
                        loss_det = torch.mean(det_pred)  # crude, adapt to detector API
                    except Exception as e:
                        print("Detector critic error:", e)
                        loss_det = torch.tensor(0.0, device=device)

                # total
                lossG = (CONFIG["lam_gan"] * loss_g_gan +
                         CONFIG["lam_L1"] * loss_l1 +
                         CONFIG["lam_perc"] * loss_perc +
                         CONFIG["lam_flow"] * loss_flow +
                         CONFIG["lam_det"] * loss_det)

            optG.zero_grad()
            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()

            pbar.set_postfix({
                "lossD": f"{lossD.item():.4f}",
                "lossG": f"{lossG.item():.4f}",
                "L1": f"{loss_l1.item():.4f}",
                "perc": f"{loss_perc.item():.4f}",
                "flow": f"{loss_flow.item() if isinstance(loss_flow, torch.Tensor) else 0.0:.4f}"
            })

        # End epoch: validate & checkpoint
        if (epoch + 1) % CONFIG["VAL_EVERY_N_EPOCHS"] == 0:
            val_metrics = validate(G, val_loader, device, vgg, l1_loss)
            print(f"VAL epoch {epoch+1}: L1={val_metrics['l1']:.4f} PSNR={val_metrics['psnr']:.2f} SSIM={val_metrics['ssim']:.4f}")
            # Save best based on L1
            cur_val = val_metrics['l1']
            ckpt_path = save_dir / f"epoch_{epoch+1}.pth"
            state = {"G":G.state_dict(), "D":D.state_dict(), "epoch":epoch+1, "config":CONFIG}
            save_checkpoint(state, ckpt_path)
            if cur_val < best_val:
                best_val = cur_val
                save_checkpoint(state, save_dir / "best_defake_model.pth")
                print("Saved new best_defake_model.pth")

    print("Training finished. Models saved to", save_dir)

# ---------- Inference: defake whole folder with overlap-aggregate ----------
def defake_folder(gen_ckpt, input_folder, out_folder, seq_len=CONFIG["SEQ_LEN"], stride=1):
    device = torch.device(CONFIG["DEVICE"])
    # build model and load weights
    G = GeneratorUNetConvLSTM(in_ch=3, base=64, hidden_ch=256).to(device)
    ck = torch.load(gen_ckpt, map_location=device)
    if "G" in ck:
        G.load_state_dict(ck["G"])
    else:
        G.load_state_dict(ck)
    G.eval()
    files = sorted([p for p in Path(input_folder).iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])
    N = len(files)
    accum = [None]*N
    counts = [0]*N
    tf = T.Compose([T.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])), T.ToTensor()])
    with torch.no_grad():
        for start in range(0, max(1, N - seq_len + 1), stride):
            seq_paths = files[start:start+seq_len]
            seq_imgs = []
            for p in seq_paths:
                im = Image.open(p).convert("RGB")
                t = tf(im)
                seq_imgs.append(t)
            seq = torch.stack(seq_imgs, dim=0).unsqueeze(0).to(device)  # (1,T,C,H,W)
            out_seq = G(seq)  # (1,T,C,H,W)
            out_seq = out_seq.squeeze(0).cpu()
            for i in range(out_seq.shape[0]):
                idx = start + i
                img_np = (out_seq[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
                if accum[idx] is None:
                    accum[idx] = img_np.astype(np.float32)
                else:
                    accum[idx] += img_np.astype(np.float32)
                counts[idx] += 1
    # finalize and save
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for i in range(N):
        if accum[i] is None:
            # if not produced (short videos), copy original
            img = Image.open(files[i]).convert("RGB").resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
            img.save(Path(out_folder)/f"frame_{i:05d}.png")
            continue
        avg = (accum[i] / max(1, counts[i])).clip(0,255).astype(np.uint8)
        Image.fromarray(avg).save(Path(out_folder)/f"frame_{i:05d}.png")
    print("Defake output saved to", out_folder)

# --------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","defake"], required=True)
    parser.add_argument("--input", help="input folder for defake mode")
    parser.add_argument("--out", help="output folder for defake mode")
    parser.add_argument("--ckpt", help="generator checkpoint for defake mode")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "defake":
        if not args.input or not args.out or not args.ckpt:
            print("For defake mode please provide --input, --out and --ckpt")
        else:
            defake_folder(args.ckpt, args.input, args.out)