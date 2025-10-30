"""
DEBUG: Test MTCNN face detection on a few sample frames
"""

import torch
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# Test on a few frames
test_frames = [
    r"H:\Celeb-real FRAMES\id0_0000\frame_00001.png",
    r"H:\Celeb-real FRAMES\id0_0001\frame_00001.png",
    r"H:\Celeb-real FRAMES\id0_0002\frame_00001.png",
]

print("Testing MTCNN face detection...")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN
detector = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=80,
    thresholds=[0.6, 0.7, 0.95],
    factor=0.709,
    post_process=True,
    device=device,
    keep_all=False,
    selection_method='largest'
)

print("MTCNN initialized")
print()

for frame_path in test_frames:
    path = Path(frame_path)
    
    if not path.exists():
        print(f"❌ Not found: {frame_path}")
        continue
    
    print(f"Testing: {path.name}")
    
    try:
        # Load image
        img_pil = Image.open(path).convert('RGB')
        print(f"  Image size: {img_pil.size}")
        
        # Detect face
        face_tensor = detector(img_pil)
        
        if face_tensor is None:
            print(f"  ❌ No face detected")
        else:
            print(f"  ✓ Face detected!")
            print(f"  Face tensor shape: {face_tensor.shape}")
            print(f"  Face tensor range: [{face_tensor.min():.2f}, {face_tensor.max():.2f}]")
            
            # Convert to numpy
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
            print(f"  NumPy shape: {face_np.shape}")
            print(f"  NumPy dtype: {face_np.dtype}")
            print(f"  NumPy range: [{face_np.min():.2f}, {face_np.max():.2f}]")
            
            # Try to save
            face_np_uint8 = face_np.astype(np.uint8)
            face_bgr = cv2.cvtColor(face_np_uint8, cv2.COLOR_RGB2BGR)
            
            output_path = f"test_face_{path.stem}.jpg"
            cv2.imwrite(output_path, face_bgr)
            print(f"  ✓ Saved to: {output_path}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

print("Test complete!")
