"""
GPU-Accelerated Face Detection with MTCNN
- Uses facenet-pytorch for real GPU acceleration
- All faces stored in single location (configurable drive)
- RAM buffering for faster writes to slow HDD
- Proper face alignment and centering
- Fixed orientation issues
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import csv
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import queue
import threading
from collections import deque
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Frame mapping CSV
FRAME_MAPPING_CSV = r"j:\DF\frame_mapping.csv"

# Output directory (ALL faces in ONE location - change to your new HDD)
OUTPUT_BASE_DIR = r"F:"  # <-- CHANGE THIS to your HDD drive letter

# Face detection settings
FACE_SIZE = 224  # Output face size (224x224 for most deepfake models)
MIN_FACE_SIZE = 80  # Minimum face size to detect (pixels)
CONFIDENCE_THRESHOLD = 0.95  # Higher threshold for better quality
FRAME_SKIP = 3  # Process every Nth frame (1=all, 2=every other, 3=every third, etc.)

# GPU settings
GPU_DEVICE = 0  # CUDA device ID
BATCH_SIZE = 32  # Larger batch for GPU efficiency

# RAM buffering settings (for slow HDD writes)
RAM_BUFFER_SIZE = 2000  # Keep up to 2000 faces in RAM before writing
WRITER_THREADS = 4  # Number of background threads writing to disk

# Resume capability
RESUME = True  # Skip already processed videos

# Output format
OUTPUT_FORMAT = "jpg"
JPEG_QUALITY = 95

# ============================================================================
# RAM BUFFERING WRITER
# ============================================================================

class BufferedWriter:
    """Write faces to disk using RAM buffer and background threads"""
    
    def __init__(self, num_threads=4, buffer_size=2000):
        self.write_queue = queue.Queue(maxsize=buffer_size)
        self.threads = []
        self.running = True
        
        # Start writer threads
        for i in range(num_threads):
            t = threading.Thread(target=self._writer_thread, daemon=True)
            t.start()
            self.threads.append(t)
        
        print(f"✓ Started {num_threads} background writer threads")
        print(f"✓ RAM buffer size: {buffer_size} faces")
    
    def _writer_thread(self):
        """Background thread that writes faces to disk"""
        while self.running:
            try:
                item = self.write_queue.get(timeout=1)
                if item is None:  # Poison pill
                    break
                
                face_data, output_path = item
                
                # Write to disk
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
        """Add face to write queue (blocks if buffer full)"""
        self.write_queue.put((face_data, output_path))
    
    def wait_completion(self):
        """Wait for all writes to complete"""
        self.write_queue.join()
    
    def shutdown(self):
        """Shutdown all writer threads"""
        self.running = False
        # Send poison pills
        for _ in self.threads:
            self.write_queue.put(None)
        # Wait for threads
        for t in self.threads:
            t.join()

# ============================================================================
# FACE DETECTOR CLASS
# ============================================================================

class FaceDetectorGPU:
    """GPU-accelerated face detector using MTCNN from facenet-pytorch"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN with GPU support
        self.detector = MTCNN(
            image_size=FACE_SIZE,
            margin=20,  # Add margin around face
            min_face_size=MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, CONFIDENCE_THRESHOLD],
            factor=0.709,
            post_process=True,  # Apply post-processing for alignment
            device=self.device,
            keep_all=False,  # Only keep best face per image
            selection_method='largest'  # Select largest face
        )
        
        print(f"✓ MTCNN initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def detect_and_align_batch(self, image_paths):
        """
        Detect and align faces from batch of images using GPU
        Returns properly aligned and centered 224x224 faces
        """
        results = []
        
        # Process each image individually to avoid batch shape mismatch
        for img_path in image_paths:
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Detect and align face on GPU
                # MTCNN returns already cropped and aligned face
                face_tensor = self.detector(img)
                
                if face_tensor is None:
                    results.append(None)
                    continue
                
                # Convert tensor to numpy (already 224x224 and aligned)
                # MTCNN returns normalized tensor in range [-1, 1], shape [C, H, W]
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                
                # Denormalize from [-1, 1] to [0, 255]
                face_np = ((face_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                
                results.append(face_bgr)
                
            except Exception as e:
                # Silently skip frames with detection errors
                results.append(None)
        
        return results

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_single_video(video_info, detector, writer, video_idx):
    """Process all frames from a single video"""
    video_name = video_info['video_name']
    frame_folder = Path(video_info['frame_folder_path'])
    video_type = video_info['type']
    
    # Create output folder structure in single location
    # Structure: OUTPUT_BASE_DIR / type / video_name
    face_folder = Path(OUTPUT_BASE_DIR) / video_type / video_name
    
    # Check if already processed
    if RESUME and face_folder.exists():
        existing_faces = len(list(face_folder.glob(f'*.{OUTPUT_FORMAT}')))
        if existing_faces > 0:
            return {
                'video_name': video_name,
                'type': video_type,
                'status': 'skipped',
                'faces_extracted': existing_faces,
                'total_frames': video_info['num_frames'],
                'face_folder': str(face_folder)
            }
    
    # Get all frame files and apply frame skip
    all_frame_files = sorted(frame_folder.glob('*.png'))
    
    # Apply frame skip (process every Nth frame)
    frame_files = all_frame_files[::FRAME_SKIP]
    
    if not frame_files:
        return {
            'video_name': video_name,
            'type': video_type,
            'status': 'no_frames',
            'faces_extracted': 0,
            'total_frames': 0,
            'frames_skipped': 0
        }
    
    faces_extracted = 0
    frames_processed = 0
    
    # Process in batches
    desc = f"[{video_idx}] {video_name}"
    with tqdm(total=len(frame_files), desc=desc, position=video_idx % 10, leave=False) as pbar:
        for i in range(0, len(frame_files), BATCH_SIZE):
            batch_files = frame_files[i:i + BATCH_SIZE]
            
            # Detect and align faces in batch on GPU
            faces = detector.detect_and_align_batch(batch_files)
            
            # Queue faces for background writing
            for frame_file, face in zip(batch_files, faces):
                frames_processed += 1
                
                if face is not None:
                    output_path = face_folder / f"{frame_file.stem}.{OUTPUT_FORMAT}"
                    
                    # Add to write queue (buffered in RAM)
                    writer.add_face(face, output_path)
                    faces_extracted += 1
                
                pbar.update(1)
    
    return {
        'video_name': video_name,
        'type': video_type,
        'status': 'completed',
        'faces_extracted': faces_extracted,
        'total_frames': frames_processed,
        'frames_skipped': len(all_frame_files) - len(frame_files),
        'total_frames': frames_processed,
        'face_folder': str(face_folder)
    }

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("=" * 80)
    print("GPU-ACCELERATED FACE DETECTION (MTCNN + RAM Buffering)")
    print("=" * 80)
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available. This script requires a GPU.")
        return
    
    # Check output directory
    output_base = Path(OUTPUT_BASE_DIR)
    if not output_base.parent.exists():
        print(f"❌ ERROR: Output drive does not exist: {OUTPUT_BASE_DIR}")
        print("Please update OUTPUT_BASE_DIR in the script to your HDD drive letter")
        return
    
    print(f"Output location: {OUTPUT_BASE_DIR}")
    print(f"All faces will be stored in ONE location")
    print()
    
    # Initialize detector
    print(f"Initializing GPU face detector...")
    detector = FaceDetectorGPU(device=f'cuda:{GPU_DEVICE}')
    print()
    
    # Initialize buffered writer
    print("Initializing RAM-buffered writer for fast HDD writes...")
    writer = BufferedWriter(num_threads=WRITER_THREADS, buffer_size=RAM_BUFFER_SIZE)
    print()
    
    # Load video list
    print(f"Loading video list from: {FRAME_MAPPING_CSV}")
    videos = []
    with open(FRAME_MAPPING_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] in ['completed', 'complete']:  # Support both status values
                videos.append(row)
    
    print(f"✓ Found {len(videos)} videos to process")
    print()
    
    # Display settings
    print(f"Settings:")
    print(f"  Face size: {FACE_SIZE}x{FACE_SIZE}")
    print(f"  Min face size: {MIN_FACE_SIZE}px")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Frame skip: Process every {FRAME_SKIP} frame(s) {'(ALL FRAMES)' if FRAME_SKIP == 1 else f'(~{100/FRAME_SKIP:.0f}% of frames)'}")
    print(f"  GPU batch size: {BATCH_SIZE}")
    print(f"  RAM buffer: {RAM_BUFFER_SIZE} faces")
    print(f"  Writer threads: {WRITER_THREADS}")
    print(f"  Output format: {OUTPUT_FORMAT.upper()}")
    print()
    
    # Process videos
    results = []
    start_time = time.time()
    
    print("Starting face detection...")
    print()
    
    with tqdm(total=len(videos), desc="Overall Progress", position=0) as main_pbar:
        for idx, video_info in enumerate(videos):
            result = process_single_video(video_info, detector, writer, idx + 1)
            results.append(result)
            main_pbar.update(1)
            
            # Update main progress bar description with stats
            completed = sum(1 for r in results if r['status'] == 'completed')
            skipped = sum(1 for r in results if r['status'] == 'skipped')
            total_faces = sum(r['faces_extracted'] for r in results)
            main_pbar.set_description(f"Overall [{completed}C {skipped}S] {total_faces:,} faces")
    
    # Wait for all writes to complete
    print()
    print("Waiting for all faces to be written to disk...")
    writer.wait_completion()
    writer.shutdown()
    
    elapsed = time.time() - start_time
    
    # Save results
    output_file = Path(FRAME_MAPPING_CSV).parent / 'face_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create face mapping CSV
    face_mapping_file = Path(FRAME_MAPPING_CSV).parent / 'face_mapping.csv'
    with open(face_mapping_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['video_name', 'type', 'face_folder', 'num_faces', 'total_frames', 'detection_rate', 'status'])
        
        for r in results:
            detection_rate = f"{r['faces_extracted']/r['total_frames']*100:.1f}%" if r['total_frames'] > 0 else "N/A"
            writer_csv.writerow([
                r['video_name'],
                r.get('type', 'unknown'),
                r.get('face_folder', ''),
                r['faces_extracted'],
                r['total_frames'],
                detection_rate,
                r['status']
            ])
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_faces = sum(r['faces_extracted'] for r in results)
    total_frames = sum(r['total_frames'] for r in results)
    total_skipped_frames = sum(r.get('frames_skipped', 0) for r in results)
    completed = sum(1 for r in results if r['status'] == 'completed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"Time elapsed: {elapsed/3600:.2f} hours")
    print(f"Videos processed: {completed}")
    print(f"Videos skipped: {skipped}")
    print(f"Frame skip setting: Every {FRAME_SKIP} frame(s)")
    print(f"Total frames processed: {total_frames:,}")
    print(f"Total frames skipped: {total_skipped_frames:,}")
    print(f"Total faces extracted: {total_faces:,}")
    print(f"Face detection rate: {total_faces/total_frames*100:.2f}%" if total_frames > 0 else "N/A")
    print(f"Processing speed: {total_frames/elapsed:.1f} frames/sec" if elapsed > 0 else "N/A")
    print(f"Time saved by skipping: ~{total_skipped_frames/total_frames*100:.1f}%" if total_frames > 0 else "N/A")
    print()
    print(f"Results saved to: {output_file}")
    print(f"Face mapping saved to: {face_mapping_file}")
    print(f"All faces stored in: {OUTPUT_BASE_DIR}")
    print()

if __name__ == "__main__":
    main()
