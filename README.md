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
- `co/framer_cpu(final)` - CPU-optimized frame extraction
- `co/face_detect_mtcnn_gpu(final).py` - GPU-accelerated face detection (MTCNN, GPU, resume, FRAME_SKIP=3)
- `co/create_mappings.py` - Video-frame relationship mapping
- `co/verify_face_extraction.py` - Verifies completeness of face extraction and reports partial/missing

## Data locations
- Faces (aligned): `F:/real/<video_name>/*.jpg`, `F:/fake/<video_name>/*.jpg`
- Frames (original): distributed across `H:/`, `I:/`, `J:/` (see `frame_mapping.csv`)
- Frame skip: 3 (affects expected counts and sampling)

## Documentation
- Architecture overview: `docs/architecture.md`
- Face detection notes: `co/FACE_DETECTION_README.md`
 
## Installation 
See `requirements.txt` for dependencies 
