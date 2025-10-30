# Deepfake Detection - Celeb-DF Dataset 
 
## Project Overview 
Deepfake detection system using the Celeb-DF dataset with GPU-accelerated preprocessing and deep learning models. 
 
## Hardware Requirements 
- GPU: NVIDIA RTX 4500 Ada (24GB VRAM) 
- CPU: Intel Xeon 72 cores 
- RAM: 256GB 
- Storage: Multiple SSDs for frame storage 
 
## Dataset 
- Celeb-DF: 590 real videos, 5,639 fake videos 
- 59 celebrities 
- 2.3M+ extracted frames 
 
## Pipeline 
1. Frame extraction (OpenCV) 
2. Face detection and alignment (MTCNN) 
3. Feature extraction 
4. Model training (CNN/ConvLSTM) 
 
## Scripts 
- `co/framer_cpu` - CPU-optimized frame extraction 
- `co/face_detect_mtcnn_gpu.py` - GPU-accelerated face detection 
- `co/create_mappings.py` - Video-frame relationship mapping 
 
## Installation 
See `requirements.txt` for dependencies 
