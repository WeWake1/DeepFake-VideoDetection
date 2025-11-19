"""
Environment Verification Script
================================
Purpose: Verify all dependencies and CUDA setup before face extraction

Author: DeepFake Detection Project
Date: November 18, 2025
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "✅ Installed"
    except ImportError:
        return False, "❌ Not installed"

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"✅ Available - {device_name} (CUDA {cuda_version}, {device_count} device(s))"
        else:
            return False, "❌ Not available - CPU only"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def main():
    print("\n" + "="*80)
    print("ENVIRONMENT VERIFICATION FOR FACE EXTRACTION")
    print("="*80)
    
    # Python version
    print(f"\n🐍 Python Version: {sys.version.split()[0]}")
    print(f"   Location: {sys.executable}")
    
    # Check critical packages
    print("\n" + "="*80)
    print("📦 REQUIRED PACKAGES")
    print("="*80)
    
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("facenet-pytorch", "facenet_pytorch"),
        ("PIL/Pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    all_installed = True
    missing_packages = []
    
    for display_name, import_name in packages:
        installed, status = check_package(display_name, import_name)
        print(f"   {display_name:20} → {status}")
        if not installed:
            all_installed = False
            missing_packages.append(display_name)
    
    # Check CUDA
    print("\n" + "="*80)
    print("🎮 GPU/CUDA STATUS")
    print("="*80)
    
    cuda_available, cuda_status = check_cuda()
    print(f"   CUDA → {cuda_status}")
    
    if cuda_available:
        import torch
        print(f"\n   PyTorch CUDA Details:")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   - GPU {i}: {props.name}")
            print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
    
    # Test MTCNN initialization
    print("\n" + "="*80)
    print("🔧 MTCNN INITIALIZATION TEST")
    print("="*80)
    
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"   Initializing MTCNN on {device}...")
        
        mtcnn = MTCNN(
            image_size=224,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.8],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        print("   ✅ MTCNN initialized successfully!")
        
        # Test with a dummy image
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8))
        result = mtcnn(dummy_img)
        print("   ✅ MTCNN test detection completed!")
        
    except Exception as e:
        print(f"   ❌ MTCNN initialization failed: {str(e)}")
        all_installed = False
    
    # Final verdict
    print("\n" + "="*80)
    print("📊 FINAL VERDICT")
    print("="*80)
    
    if all_installed and cuda_available:
        print("\n   ✅ READY TO START FACE EXTRACTION!")
        print("   All dependencies installed and CUDA is available.")
        print("\n   You can now run:")
        print("   python preprocessing\\extract_faces_dfd.py")
        print("   python preprocessing\\extract_faces_ff.py")
    else:
        print("\n   ⚠️ ENVIRONMENT NOT READY")
        if missing_packages:
            print(f"\n   Missing packages: {', '.join(missing_packages)}")
            print("\n   Install missing packages with:")
            print(f"   pip install {' '.join(missing_packages)}")
        
        if not cuda_available:
            print("\n   ⚠️ CUDA not available - face extraction will be VERY SLOW on CPU")
            print("   Make sure you have:")
            print("   1. NVIDIA GPU driver installed")
            print("   2. CUDA toolkit installed")
            print("   3. PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
