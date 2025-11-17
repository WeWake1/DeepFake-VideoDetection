"""
GPU Readiness Check - Verify CUDA and GPU before running extraction/detection
"""
import torch
import sys

print("="*70)
print("🔍 GPU READINESS CHECK")
print("="*70)

# PyTorch and CUDA
print(f"\n📦 PyTorch Version: {torch.__version__}")
print(f"🔧 CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ ERROR: CUDA not available!")
    print("   Please check your PyTorch installation.")
    sys.exit(1)

print(f"🔧 CUDA Version: {torch.version.cuda}")
print(f"🔧 cuDNN Version: {torch.backends.cudnn.version()}")

# GPU Details
gpu_count = torch.cuda.device_count()
print(f"\n🎮 GPU Count: {gpu_count}")

for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\n🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    print(f"   Multi-Processors: {props.multi_processor_count}")
    
    # Memory status
    torch.cuda.set_device(i)
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
    
    print(f"   Memory Allocated: {allocated:.2f} GB")
    print(f"   Memory Reserved: {reserved:.2f} GB")
    print(f"   Memory Free: {free:.2f} GB")

# Test GPU with small tensor
print("\n🧪 Testing GPU computation...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print("✅ GPU computation successful!")
except Exception as e:
    print(f"❌ GPU computation failed: {e}")
    sys.exit(1)

# Check facenet-pytorch (MTCNN)
print("\n📦 Checking facenet-pytorch...")
try:
    from facenet_pytorch import MTCNN
    detector = MTCNN(device='cuda:0')
    print("✅ MTCNN initialized successfully on GPU")
except Exception as e:
    print(f"❌ MTCNN initialization failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL CHECKS PASSED - GPU READY FOR PROCESSING")
print("="*70)
