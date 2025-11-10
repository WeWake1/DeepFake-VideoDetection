import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
import torchvision.transforms as T

from train_all_in_one import TwoStreamDetector

def find_videos(frames_root):
    fake, real = [], []
    for root, _, files in os.walk(frames_root):
        if any(f.lower().endswith(('.jpg','.jpeg','.png','.bmp')) for f in files):
            if 'fake' in root.lower():
                fake.append(root)
            elif 'real' in root.lower():
                real.append(root)
    return fake, real

def pick_frame(video_dir):
    files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    files.sort()
    if not files:
        raise RuntimeError(f'No images in {video_dir}')
    return os.path.join(video_dir, files[len(files)//2])

def build_seq_from_frame(path, seq_len, size, device):
    img = Image.open(path).convert('RGB').resize((size, size))
    tf = T.Compose([T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    t = tf(img)  # C,H,W
    seq = t.unsqueeze(0).repeat(seq_len, 1, 1, 1)  # T,C,H,W
    return seq.unsqueeze(0).to(device)  # 1,T,C,H,W


def build_seq_from_pil(img: Image.Image, seq_len, size, device):
    img = img.convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size))
    tf = T.Compose([T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    t = tf(img)
    seq = t.unsqueeze(0).repeat(seq_len, 1, 1, 1)
    return seq.unsqueeze(0).to(device)


def histogram_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    # Simple per-channel histogram matching (source and template are HWC uint8 or float)
    source = source.astype(np.float32)
    template = template.astype(np.float32)
    matched = np.zeros_like(source)
    for ch in range(source.shape[2]):
        s = source[:, :, ch].ravel()
        t = template[:, :, ch].ravel()
        s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(t, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        matched[..., ch] = interp_t_values[bin_idx].reshape(source.shape[0], source.shape[1])
    return matched.astype(source.dtype)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames-dir', default='Data')
    p.add_argument('--checkpoint', default=os.path.join('checkpoints','best_model.pth'))
    p.add_argument('--use-flow', action='store_true')
    p.add_argument('--flows-dir', default=None)
    p.add_argument('--same-image', action='store_true', help='Use the same selected frame for both fake and real')
    p.add_argument('--image', default=None, help='Explicit image path to use for fake; use with --match-real to find matching real by basename')
    p.add_argument('--match-real', action='store_true', help='When --image is given, locate a real image with the same filename basename')
    p.add_argument('--seq-len', type=int, default=8)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    fake_dirs, real_dirs = find_videos(args.frames_dir)
    if not fake_dirs or not real_dirs:
        print("Need at least one fake and one real video folder under", args.frames_dir); return

    vf = random.choice(fake_dirs)
    vr = random.choice(real_dirs)
    pf = pick_frame(vf)
    pr = pick_frame(vr)
    # allow forcing the same image for both sides (or explicit image path)
    if args.image:
        if not os.path.exists(args.image):
            print("Image not found:", args.image)
            return
        pf = args.image
        if args.match_real:
            # try to find a matching basename in real_dirs
            base = os.path.basename(args.image)
            found = None
            for rd in real_dirs:
                cand = os.path.join(rd, base)
                if os.path.exists(cand):
                    found = cand
                    break
            if found:
                pr = found
                print("Found matching real for basename", base, "->", pr)
            else:
                pr = pf
                print("No matching real found for basename", base, "— using the same image for both sides")
        else:
            pr = pf
            print("Using explicit image for both fake and real:", args.image)
    elif args.same_image:
        pr = pf
        print("Using same image for fake and real:", pf)
    else:
        print("Fake frame:", pf)
        print("Real frame:", pr)

    device = torch.device(args.device)
    model = TwoStreamDetector(use_flow=args.use_flow).to(device)
    if os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print("Loaded checkpoint:", args.checkpoint)
    else:
        print("Checkpoint not found, using untrained model:", args.checkpoint)

    model.eval()
    seq_f = build_seq_from_frame(pf, args.seq_len, args.size, device)
    seq_r = build_seq_from_frame(pr, args.seq_len, args.size, device)

    if args.use_flow:
        flow_f = torch.zeros(1, max(1, args.seq_len-1), 2, args.size, args.size, device=device)
        flow_r = flow_f.clone()
    else:
        flow_f = flow_r = None

    with torch.no_grad():
        out_f = torch.sigmoid(model(seq_f, flow_f)).item()
        out_r = torch.sigmoid(model(seq_r, flow_r)).item()

    realness_f = 1.0 - out_f
    realness_r = 1.0 - out_r

    print(f"Fake GT=1  Pred Fake={out_f:.4f}  Pred Real={realness_f:.4f}")
    print(f"Real GT=0  Pred Fake={out_r:.4f}  Pred Real={realness_r:.4f}")

    img_f = Image.open(pf).convert('RGB').resize((args.size,args.size))
    img_r = Image.open(pr).convert('RGB').resize((args.size,args.size))
    W = args.size*2 + 10
    H = args.size + 60
    out = Image.new('RGB', (W,H), (255,255,255))
    out.paste(img_f, (5,5))
    out.paste(img_r, (args.size+10,5))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((10, args.size+10), f"GT=FAKE    PredFake={out_f:.4f}    PredReal={realness_f:.4f}", fill=(255,0,0), font=font)
    draw.text((args.size+15, args.size+10), f"GT=REAL    PredFake={out_r:.4f}    PredReal={realness_r:.4f}", fill=(0,128,0), font=font)

    # create a defaked variant by histogram-matching fake -> real
    try:
        fa = np.array(img_f).astype(np.uint8)
        ra = np.array(img_r).astype(np.uint8)
        matched = histogram_match(fa, ra).clip(0,255).astype(np.uint8)
        img_def = Image.fromarray(matched)
    except Exception:
        img_def = img_f.copy()

    # run model on defaked image
    seq_def = build_seq_from_pil(img_def, args.seq_len, args.size, device)
    with torch.no_grad():
        out_def = torch.sigmoid(model(seq_def, None)).item()

    reduction = out_f - out_def
    reduction_pct = (reduction / out_f * 100.0) if out_f > 1e-6 else 0.0

    # compose 3-panel image: fake | defaked | real
    W3 = args.size * 3 + 20
    H3 = args.size + 80
    out3 = Image.new('RGB', (W3, H3), (255,255,255))
    out3.paste(img_f, (5,5))
    out3.paste(img_def, (args.size+10,5))
    out3.paste(img_r, (args.size*2+15,5))
    draw3 = ImageDraw.Draw(out3)
    draw3.text((10, args.size+10), f"Orig Fake: fake_prob={out_f:.4f} real_prob={realness_f:.4f}", fill=(255,0,0), font=font)
    draw3.text((args.size+15, args.size+10), f"Defaked: fake_prob={out_def:.4f} real_prob={(1-out_def):.4f}  delta={reduction:.4f} ({reduction_pct:.1f}%)", fill=(0,0,128), font=font)
    draw3.text((args.size*2+20, args.size+10), f"Real: fake_prob={out_r:.4f} real_prob={realness_r:.4f}", fill=(0,128,0), font=font)

    out_path = os.path.join('checkpoints','realness_result.png')
    out.save(out_path)
    out3_path = os.path.join('checkpoints','realness_defake_result.png')
    out3.save(out3_path)
    print("Saved:", out_path)
    print("Saved:", out3_path)
    try:
        os.startfile(out3_path)
    except Exception:
        pass

if __name__ == '__main__':
    main()