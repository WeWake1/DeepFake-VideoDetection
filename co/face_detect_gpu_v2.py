"""
GPU-Accelerated Face Detection, Cropping, and Alignment
Uses facenet-pytorch's MTCNN for GPU acceleration
Processes frames from extracted videos with real GPU support
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Frame mapping CSV
FRAME_MAPPING_CSV = r"j:\DF\frame_mapping.csv"

# Face detection settings
FACE_SIZE = 224  # Output face size (224x224 for most deepfake models)
MIN_FACE_SIZE = 80  # Minimum face size to detect (pixels)
CONFIDENCE_THRESHOLD = 0.9  # Detection confidence threshold

# GPU settings
GPU_DEVICE = 0  # CUDA device ID
BATCH_SIZE = 16  # Process frames in batches for GPU efficiency

# Processing settings
IO_THREADS = 16  # Threads for I/O operations

# Resume capability
RESUME = True  # Skip already processed videos

# Output format
OUTPUT_FORMAT = "jpg"  # jpg or png
JPEG_QUALITY = 95  # JPEG quality (1-100)

# ============================================================================
# FACE DETECTOR CLASS
# ============================================================================

class FaceDetectorGPU:
    """GPU-accelerated face detector using MTCNN"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN with GPU support
        self.detector = MTCNN(
            image_size=FACE_SIZE,
            margin=0,
            min_face_size=MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, CONFIDENCE_THRESHOLD],
            device=self.device,
            keep_all=False,  # Only keep the largest/most confident face
            post_process=False  # We'll do our own alignment
        )
        
        print(f"✓ Face detector initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def align_face(self, image, left_eye, right_eye):
        """Align face based on eye positions"""
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                      (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Perform rotation
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def detect_and_crop_batch(self, image_paths):
        """Detect and crop faces from a batch of images using GPU"""
        results = []
        
        # Load images
        images = []
        valid_indices = []
        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                results.append(None)
        
        if not images:
            return results
        
        # Detect faces in batch on GPU
        try:
            batch_boxes, batch_probs, batch_landmarks = self.detector.detect(images, landmarks=True)
            
            for idx, (boxes, probs, landmarks) in enumerate(zip(batch_boxes, batch_probs, batch_landmarks)):
                original_idx = valid_indices[idx]
                
                if boxes is None or probs is None:
                    # Insert None at correct position
                    while len(results) <= original_idx:
                        results.append(None)
                    continue
                
                # Convert PIL to numpy
                img_np = np.array(images[idx])
                
                # Get the most confident face
                best_idx = 0  # MTCNN already returns best face when keep_all=False
                box = boxes[best_idx] if len(boxes.shape) > 1 else boxes
                landmark = landmarks[best_idx] if len(landmarks.shape) > 2 else landmarks
                
                # Extract face region with margin
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Add margin
                h, w = img_np.shape[:2]
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face = img_np[y1:y2, x1:x2]
                
                # Align if landmarks available
                if landmark is not None and len(landmark) >= 2:
                    # Landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
                    left_eye = landmark[0] - [x1, y1]
                    right_eye = landmark[1] - [x1, y1]
                    face = self.align_face(face, left_eye, right_eye)
                
                # Resize to target size
                face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
                
                # Insert at correct position
                while len(results) <= original_idx:
                    results.append(None)
                results[original_idx] = face
                
        except Exception as e:
            print(f"Batch detection error: {e}")
            # Fill remaining with None
            while len(results) < len(image_paths):
                results.append(None)
        
        return results

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_single_video(video_info, detector, pbar_position):
    """Process all frames from a single video"""
    video_name = video_info['video_name']
    frame_folder = Path(video_info['frame_folder_path'])
    
    # Create output folder (same location, just replace FRAMES with FACES)
    face_folder = Path(str(frame_folder).replace(' FRAMES', ' FACES'))
    
    # Check if already processed
    if RESUME and face_folder.exists():
        existing_faces = len(list(face_folder.glob(f'*.{OUTPUT_FORMAT}')))
        if existing_faces > 0:
            return {
                'video_name': video_name,
                'status': 'skipped',
                'faces_extracted': existing_faces,
                'total_frames': video_info['num_frames']
            }
    
    # Create output directory
    face_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(frame_folder.glob('*.png'))
    
    if not frame_files:
        return {
            'video_name': video_name,
            'status': 'no_frames',
            'faces_extracted': 0,
            'total_frames': 0
        }
    
    faces_extracted = 0
    frames_processed = 0
    
    # Process in batches
    with tqdm(total=len(frame_files), desc=video_name, position=pbar_position, leave=False) as pbar:
        for i in range(0, len(frame_files), BATCH_SIZE):
            batch_files = frame_files[i:i + BATCH_SIZE]
            
            # Detect faces in batch
            faces = detector.detect_and_crop_batch(batch_files)
            
            # Save detected faces
            for frame_file, face in zip(batch_files, faces):
                frames_processed += 1
                
                if face is not None:
                    # Generate output filename
                    output_file = face_folder / f"{frame_file.stem}.{OUTPUT_FORMAT}"
                    
                    # Save face
                    if OUTPUT_FORMAT == 'jpg':
                        cv2.imwrite(str(output_file), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    else:
                        cv2.imwrite(str(output_file), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    
                    faces_extracted += 1
                
                pbar.update(1)
    
    return {
        'video_name': video_name,
        'status': 'completed',
        'faces_extracted': faces_extracted,
        'total_frames': frames_processed,
        'face_folder': str(face_folder)
    }

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("=" * 80)
    print("GPU-ACCELERATED FACE DETECTION AND ALIGNMENT (facenet-pytorch)")
    print("=" * 80)
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available. This script requires a GPU.")
        return
    
    print(f"Initializing GPU face detector...")
    detector = FaceDetectorGPU(device=f'cuda:{GPU_DEVICE}')
    print()
    
    # Load video list
    print(f"Loading video list from: {FRAME_MAPPING_CSV}")
    videos = []
    with open(FRAME_MAPPING_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'completed':
                videos.append(row)
    
    print(f"✓ Found {len(videos)} videos to process")
    print()
    
    # Display settings
    print(f"Face size: {FACE_SIZE}x{FACE_SIZE}")
    print(f"Min face size: {MIN_FACE_SIZE}px")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output format: {OUTPUT_FORMAT.upper()}")
    print()
    
    # Process videos sequentially (GPU handles parallelism internally)
    results = []
    
    with tqdm(total=len(videos), desc="Overall Progress", position=0) as main_pbar:
        for video_info in videos:
            result = process_single_video(video_info, detector, pbar_position=1)
            results.append(result)
            main_pbar.update(1)
    
    # Save results
    output_file = Path(FRAME_MAPPING_CSV).parent / 'face_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_faces = sum(r['faces_extracted'] for r in results)
    total_frames = sum(r['total_frames'] for r in results)
    completed = sum(1 for r in results if r['status'] == 'completed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"Videos processed: {completed}")
    print(f"Videos skipped: {skipped}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total faces extracted: {total_faces:,}")
    print(f"Success rate: {total_faces/total_frames*100:.2f}%" if total_frames > 0 else "N/A")
    print(f"\nResults saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
