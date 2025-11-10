"""Transform & evaluate script for video sequences

Loads a sequence of frames from fake and real video folders, applies transforms to the entire sequence,
runs the trained detector on each transformed sequence, and reports the best transform that minimizes fake probability.

Usage (cmd.exe):
  python transform_eval_sequence.py --fake-video-folder Data\fake\fake\id20_id3_0001 --real-video-folder Data\real\real\id20_0000 --checkpoint checkpoints\best_model.pth

The script saves results in the out-dir.
"""
from __future__ import annotations
import argparse
import os
import json
from typing import List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from train_all_in_one import TwoStreamDetector


def load_image(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    return img


def jpeg_compress(img: Image.Image, quality: int = 70) -> Image.Image:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def median_filter(img: Image.Image, size: int = 3) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(size=size))


def color_jitter(img: Image.Image, brightness=1.05, contrast=1.05, saturation=1.05) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img


def sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))


def histogram_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
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


def gamma_correction(img: Image.Image, gamma: float = 1.0) -> Image.Image:
    if gamma == 1.0:
        return img
    inv_gamma = 1.0 / gamma
    lut = [pow(i / 255., inv_gamma) * 255 for i in range(256)]
    lut = [int(max(0, min(255, x))) for x in lut]
    return img.point(lut * 3)


def hist_equalize(img: Image.Image) -> Image.Image:
    try:
        import cv2
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(arr)
        cla = cv2.equalizeHist(l)
        merged = cv2.merge((cla, a, b))
        res = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(res)
    except Exception:
        try:
            from PIL import ImageOps
            r, g, b = img.split()
            return Image.merge('RGB', (ImageOps.equalize(r), ImageOps.equalize(g), ImageOps.equalize(b)))
        except Exception:
            return img


def clahe(img: Image.Image, clip=2.0, tile_grid_size=(8,8)) -> Image.Image:
    try:
        import cv2
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(arr)
        clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid_size)
        cl = clahe_obj.apply(l)
        merged = cv2.merge((cl, a, b))
        res = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(res)
    except Exception:
        return img


def pil_to_tensor(img: Image.Image):
    tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    return tf(img)


def load_sequence(folder: str, seq_len: int, size: int) -> List[Image.Image]:
    frames = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            frames.append(load_image(os.path.join(folder, f), size))
            if len(frames) >= seq_len:
                break
    return frames[:seq_len]


def make_sequence_tensor(seq: List[Image.Image]) -> torch.Tensor:
    tensors = [pil_to_tensor(img) for img in seq]
    seq_tensor = torch.stack(tensors, dim=0)  # (T, C, H, W)
    return seq_tensor.unsqueeze(0)  # (1, T, C, H, W)


def run_model_on_sequence(model: TwoStreamDetector, seq: List[Image.Image], device: torch.device) -> float:
    seq_tensor = make_sequence_tensor(seq).to(device)
    with torch.no_grad():
        logits = model(seq_tensor, None)
        prob = float(torch.sigmoid(logits).cpu().item())
    return prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fake-video-folder', required=True, help='Folder with fake video frames')
    p.add_argument('--real-video-folder', required=False, help='Folder with real video frames (optional, defaults to matching)')
    p.add_argument('--checkpoint', default=os.path.join('checkpoints','best_model.pth'))
    p.add_argument('--seq-len', type=int, default=8)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='transform_eval_sequence_out')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    model = TwoStreamDetector(use_flow=False).to(device)
    if os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print('Loaded checkpoint:', args.checkpoint)
    else:
        print('Checkpoint not found, using random-initialized model')
    model.eval()

    # Load fake sequence
    fake_seq = load_sequence(args.fake_video_folder, args.seq_len, args.size)
    if len(fake_seq) < args.seq_len:
        print(f'Not enough frames in fake folder, got {len(fake_seq)}')
        return

    # Load real sequence
    if not args.real_video_folder:
        # Parse fake folder to get id
        folder_name = os.path.basename(args.fake_video_folder)
        if '_' in folder_name:
            ids = folder_name.split('_')[:2]
            ids = [id.replace('id', '') for id in ids]
            real_folder = f'id{ids[0]}_0000'
            args.real_video_folder = os.path.join('Data/real/real', real_folder)
            if not os.path.exists(args.real_video_folder):
                print('Matching real folder not found')
                return
        else:
            print('Cannot parse id from fake folder')
            return
    real_seq = load_sequence(args.real_video_folder, args.seq_len, args.size)
    if len(real_seq) < args.seq_len:
        print(f'Not enough frames in real folder, got {len(real_seq)}')
        return

    transforms = {
        'original': lambda seq: seq,
        'jpeg_q90': lambda seq: [jpeg_compress(img, 90) for img in seq],
        'jpeg_q70': lambda seq: [jpeg_compress(img, 70) for img in seq],
        'jpeg_q50': lambda seq: [jpeg_compress(img, 50) for img in seq],
        'gaussian_blur_r0.5': lambda seq: [gaussian_blur(img, 0.5) for img in seq],
        'gaussian_blur_r1.0': lambda seq: [gaussian_blur(img, 1.0) for img in seq],
        'gaussian_blur_r2.0': lambda seq: [gaussian_blur(img, 2.0) for img in seq],
        'median_3': lambda seq: [median_filter(img, 3) for img in seq],
        'median_5': lambda seq: [median_filter(img, 5) for img in seq],
        'color_jitter_l1.02': lambda seq: [color_jitter(img, 1.02, 1.02, 1.02) for img in seq],
        'color_jitter_l1.05': lambda seq: [color_jitter(img, 1.05, 1.05, 1.05) for img in seq],
        'sharpen': lambda seq: [sharpen(img) for img in seq],
        'gamma_0.8': lambda seq: [gamma_correction(img, 0.8) for img in seq],
        'gamma_0.9': lambda seq: [gamma_correction(img, 0.9) for img in seq],
        'gamma_1.1': lambda seq: [gamma_correction(img, 1.1) for img in seq],
        'gamma_1.2': lambda seq: [gamma_correction(img, 1.2) for img in seq],
        'hist_equalize': lambda seq: [hist_equalize(img) for img in seq],
        'clahe_clip2': lambda seq: [clahe(img, 2.0) for img in seq],
        'clahe_clip4': lambda seq: [clahe(img, 4.0) for img in seq],
    }

    # Add hist_match if real available
    def hist_match_seq(seq: List[Image.Image]) -> List[Image.Image]:
        return [Image.fromarray(histogram_match(np.array(img), np.array(real_seq[i])).clip(0,255).astype(np.uint8)) for i, img in enumerate(seq)]
    transforms['hist_match_to_real'] = hist_match_seq

    results = []
    for name, fn in transforms.items():
        tseq = fn(fake_seq)
        prob = run_model_on_sequence(model, tseq, device)
        results.append({'transform': name, 'fake_prob': prob, 'real_prob': 1.0 - prob})
        print(f'{name}: fake_prob={prob:.4f}')

    # Plot results
    transform_names = [r['transform'] for r in results]
    fake_probs = [r['fake_prob'] for r in results]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(transform_names)), fake_probs, tick_label=transform_names)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Fake Probability')
    plt.title('Fake Probability by Transform (Lower is Better)')
    plt.tight_layout()
    # Highlight original
    orig_idx = transform_names.index('original')
    bars[orig_idx].set_color('red')
    bars[orig_idx].set_label('Original')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'results.png'))
    plt.close()

    # Save JSON
    json_path = os.path.join(args.out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Find best
    orig = next(r for r in results if r['transform'] == 'original')
    candidates = [r for r in results if r['transform'] != 'original']
    if candidates:
        best = min(candidates, key=lambda r: r['fake_prob'])
        delta = orig['fake_prob'] - best['fake_prob']
        reduction_pct = (delta / orig['fake_prob'] * 100.0) if orig['fake_prob'] > 1e-6 else 0.0
        print(f"Best transform: {best['transform']} reduced fake_prob {orig['fake_prob']:.4f} -> {best['fake_prob']:.4f} (delta={delta:.4f}, {reduction_pct:.1f}%)")

        # Save best sequence frames
        best_seq = transforms[best['transform']](fake_seq)
        for i, img in enumerate(best_seq[:3]):  # Save first 3
            img.save(os.path.join(args.out_dir, f'best_frame_{i}.png'))
    else:
        print('No transforms to compare')

    print('Wrote outputs to', args.out_dir)


if __name__ == '__main__':
    main()