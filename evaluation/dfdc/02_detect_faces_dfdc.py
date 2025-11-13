"""
Face Detector for DFDC Dataset - Evaluation Version
Adapted from preprocessing/face_detect_mtcnn_gpu(final).py

Detects and aligns faces from DFDC frames using GPU-accelerated MTCNN.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import queue
import threading

# ===== DFDC CONFIGURATION =====
INPUT_FRAMES_DIR = r"H:\EVAL-1\frames"
OUTPUT_FACES_DIR = r"F:\EVAL - 1 (FRAMES)"

# Face detection settings
FACE_SIZE = 224
MIN_FACE_SIZE = 80
CONFIDENCE_THRESHOLD = 0.95
FRAME_SKIP = 1  # Process every 3rd frame (matching training)

# GPU settings
GPU_DEVICE = 0
BATCH_SIZE = 32

# RAM buffering
RAM_BUFFER_SIZE = 2000
WRITER_THREADS = 4

# Output format
OUTPUT_FORMAT = "jpg"
JPEG_QUALITY = 95

# Resume capability
RESUME = True

# ============================================================================
# RAM BUFFERING WRITER
# ============================================================================

class BufferedWriter:
    """Write faces to disk using RAM buffer and background threads"""
    
    def __init__(self, num_threads=4, buffer_size=2000):
        self.write_queue = queue.Queue(maxsize=buffer_size)
        self.threads = []
        self.running = True
        
        for i in range(num_threads):
            t = threading.Thread(target=self._writer_thread, daemon=True)
            t.start()
            self.threads.append(t)
        
        print(f"✓ Started {num_threads} background writer threads")
    
    def _writer_thread(self):
        """Background thread that writes faces to disk"""
        while self.running:
            try:
                item = self.write_queue.get(timeout=1)
                if item is None:
                    break
                
                face_data, output_path = item
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if OUTPUT_FORMAT == 'jpg':
                    cv2.imwrite(str(output_path), face_data,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                else:
                    cv2.imwrite(str(output_path), face_data)
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Write error: {e}")
    
    def add_face(self, face_data, output_path):
        """Add face to write queue"""
        self.write_queue.put((face_data, output_path))
    
    def wait_completion(self):
        """Wait for all writes to complete"""
        self.write_queue.join()
    
    def shutdown(self):
        """Shutdown all writer threads"""
        self.running = False
        for _ in self.threads:
            self.write_queue.put(None)
        for t in self.threads:
            t.join()

# ============================================================================
# FACE DETECTOR CLASS
# ============================================================================

class FaceDetectorGPU:
    """GPU-accelerated face detector using MTCNN"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.detector = MTCNN(
            image_size=FACE_SIZE,
            margin=20,
            min_face_size=MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, CONFIDENCE_THRESHOLD],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False,
            selection_method='largest'
        )
        
        print(f"✓ MTCNN initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    def detect_and_align_batch(self, image_paths):
        """Detect and align faces from batch of images"""
        faces = []
        successful_paths = []
        
        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception:
                images.append(None)
        
        # Detect faces
        for img, img_path in zip(images, image_paths):
            if img is None:
                faces.append(None)
                continue
            
            try:
                face_tensor = self.detector(img)
                
                if face_tensor is not None:
                    # Convert to numpy
                    face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                    face_np = (face_np * 128 + 127.5).astype(np.uint8)
                    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    faces.append(face_bgr)
                    successful_paths.append(img_path)
                else:
                    faces.append(None)
            except Exception:
                faces.append(None)
        
        return faces, successful_paths

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_video_folder(video_folder, output_base, detector, writer, stats):
    """Process all frames from a single video folder"""
    video_name = video_folder.name
    output_folder = output_base / video_name
    
    # Check if already processed
    if RESUME and output_folder.exists():
        existing_faces = list(output_folder.glob(f'*.{OUTPUT_FORMAT}'))
        if len(existing_faces) > 5:  # Skip if has some faces
            stats['skipped'] += 1
            return f"⏭️  {video_name} - already processed ({len(existing_faces)} faces)"
    
    # Get all frame files (with frame skip)
    all_frames = sorted(list(video_folder.glob('*.png')))
    
    if not all_frames:
        stats['no_frames'] += 1
        return f"⚠️  {video_name} - no frames found"
    
    # Apply frame skip (every 3rd frame)
    frames_to_process = all_frames[::FRAME_SKIP]
    
    # Process in batches
    faces_detected = 0
    
    for i in range(0, len(frames_to_process), BATCH_SIZE):
        batch = frames_to_process[i:i+BATCH_SIZE]
        
        faces, successful_paths = detector.detect_and_align_batch(batch)
        
        # Write detected faces
        for face, frame_path in zip(faces, successful_paths):
            if face is not None:
                frame_name = frame_path.stem
                output_path = output_folder / f"{frame_name}.{OUTPUT_FORMAT}"
                writer.add_face(face, output_path)
                faces_detected += 1
    
    if faces_detected == 0:
        stats['no_faces'] += 1
        return f"⚠️  {video_name} - no faces detected"
    else:
        stats['processed'] += 1
        stats['total_faces'] += faces_detected
        return f"✅ {video_name} - {faces_detected} faces"

def main():
    """Main face detection logic for DFDC"""
    print("="*70)
    print("🎯 DFDC FACE DETECTOR (EVALUATION)")
    print("="*70)
    print(f"   Input: {INPUT_FRAMES_DIR}")
    print(f"   Output: {OUTPUT_FACES_DIR}")
    print(f"   Face Size: {FACE_SIZE}x{FACE_SIZE}")
    print(f"   Frame Skip: Every {FRAME_SKIP} frames")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   GPU Device: cuda:{GPU_DEVICE}")
    print("="*70)
    
    input_dir = Path(INPUT_FRAMES_DIR)
    output_dir = Path(OUTPUT_FACES_DIR)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all video folders
    video_folders = [f for f in input_dir.iterdir() if f.is_dir()]
    
    if not video_folders:
        print("❌ No video folders found.")
        return
    
    print(f"Found {len(video_folders)} videos to process.\n")
    
    # Initialize detector and writer
    detector = FaceDetectorGPU(device=f'cuda:{GPU_DEVICE}')
    writer = BufferedWriter(num_threads=WRITER_THREADS, buffer_size=RAM_BUFFER_SIZE)
    
    stats = {
        'processed': 0,
        'skipped': 0,
        'no_faces': 0,
        'no_frames': 0,
        'total_faces': 0
    }
    
    # Process all videos
    try:
        with tqdm(total=len(video_folders), desc="Detecting Faces", unit="video") as pbar:
            for video_folder in video_folders:
                result_msg = process_video_folder(video_folder, output_dir, detector, writer, stats)
                tqdm.write(result_msg)
                pbar.update(1)
        
        # Wait for all writes to complete
        print("\n⏳ Waiting for background writes to complete...")
        writer.wait_completion()
        
    finally:
        writer.shutdown()
    
    print("\n" + "="*70)
    print("📊 DETECTION SUMMARY")
    print("="*70)
    print(f"   Processed: {stats['processed']}")
    print(f"   Skipped (already done): {stats['skipped']}")
    print(f"   No faces detected: {stats['no_faces']}")
    print(f"   No frames found: {stats['no_frames']}")
    print(f"   Total faces extracted: {stats['total_faces']:,}")
    print("="*70)
    print(f"📁 Faces saved to: {OUTPUT_FACES_DIR}")

if __name__ == "__main__":
    main()
