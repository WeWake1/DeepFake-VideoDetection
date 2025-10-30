"""
GPU-Accelerated Face Detection, Cropping, and Alignment
Uses RetinaFace for maximum accuracy in deepfake detection
Processes frames from extracted videos with GPU acceleration
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import csv
from concurrent.futures import ThreadPoolExecutor
import torch
from retinaface import RetinaFace

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Frame mapping CSV
FRAME_MAPPING_CSV = r"j:\DF\frame_mapping.csv"

# Output directories (will distribute across drives)
OUTPUT_DIRS = [
    r"H:\Celeb-DF FACES",
    r"J:\DF\Celeb-DF FACES", 
    r"I:\Celeb-DF FACES"
]

# Face detection settings
FACE_SIZE = 224  # Output face size (224x224 for most deepfake models)
MIN_FACE_SIZE = 80  # Minimum face size to detect (pixels)
CONFIDENCE_THRESHOLD = 0.9  # RetinaFace confidence threshold

# GPU settings
GPU_DEVICE = 0  # CUDA device ID
BATCH_SIZE = 32  # Process frames in batches for GPU efficiency

# Processing settings
CONCURRENT_VIDEOS = 8  # Process multiple videos in parallel
IO_THREADS = 32  # Threads for I/O operations per video

# Resume capability
RESUME = True  # Skip already processed videos

# Output format
OUTPUT_FORMAT = "jpg"  # jpg or png
JPEG_QUALITY = 95  # JPEG quality (1-100)

# ============================================================================
# FACE DETECTION AND ALIGNMENT
# ============================================================================

class FaceDetector:
    """GPU-accelerated face detector using RetinaFace"""
    
    def __init__(self, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if not torch.cuda.is_available():
            print("⚠️ WARNING: CUDA not available! Using CPU (will be slow)")
        else:
            print(f"✓ GPU detected: {torch.cuda.get_device_name(gpu_id)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    
    def align_face(self, img, facial_area, left_eye, right_eye):
        """
        Align face based on eye positions (essential for deepfake detection)
        """
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                      (left_eye[1] + right_eye[1]) // 2)
        
        # Rotate image to align eyes horizontally
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Crop face region from aligned image
        x, y, w, h = facial_area
        face_crop = aligned[y:y+h, x:x+w]
        
        return face_crop
    
    def detect_and_align(self, frame):
        """
        Detect face in frame and return aligned, cropped face
        Returns: (face_image, confidence) or (None, 0) if no face detected
        """
        try:
            # Detect faces with RetinaFace
            faces = RetinaFace.detect_faces(frame)
            
            if not isinstance(faces, dict) or len(faces) == 0:
                return None, 0.0
            
            # Get the face with highest confidence
            best_face = None
            best_conf = 0.0
            
            for key, face_data in faces.items():
                conf = face_data.get('score', 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_face = face_data
            
            if best_face is None or best_conf < CONFIDENCE_THRESHOLD:
                return None, 0.0
            
            # Extract facial area and landmarks
            facial_area = best_face['facial_area']
            landmarks = best_face['landmarks']
            
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # Check minimum face size
            x, y, w, h = facial_area
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                return None, 0.0
            
            # Align face based on eyes
            aligned_face = self.align_face(frame, facial_area, left_eye, right_eye)
            
            # Resize to standard size
            face_resized = cv2.resize(aligned_face, (FACE_SIZE, FACE_SIZE), 
                                     interpolation=cv2.INTER_LANCZOS4)
            
            return face_resized, best_conf
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None, 0.0


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def get_output_dir(video_idx):
    """Get output directory for this video (round-robin across drives)"""
    return OUTPUT_DIRS[video_idx % len(OUTPUT_DIRS)]


def get_resume_info(output_path):
    """Check if video already processed"""
    if not RESUME:
        return False, 0
    
    if not os.path.exists(output_path):
        return False, 0
    
    # Count existing faces
    existing_faces = [f for f in os.listdir(output_path) 
                     if f.endswith(f'.{OUTPUT_FORMAT}')]
    
    if len(existing_faces) > 0:
        return True, len(existing_faces)
    
    return False, 0


def process_single_video(video_info, video_idx, detector):
    """
    Process one video: detect faces in all frames
    """
    video_name = video_info['video_name']
    frame_folder = video_info['frame_folder_path']
    video_type = video_info['type']
    
    # Skip if frames missing
    if not os.path.exists(frame_folder):
        return {
            'video_name': video_name,
            'status': 'frames_missing',
            'faces_detected': 0,
            'frames_processed': 0
        }
    
    # Determine output directory
    output_dir = get_output_dir(video_idx)
    output_path = os.path.join(output_dir, f"{video_type} FACES", video_name)
    
    # Check if already processed
    already_done, existing_count = get_resume_info(output_path)
    if already_done:
        return {
            'video_name': video_name,
            'status': 'already_processed',
            'faces_detected': existing_count,
            'frames_processed': existing_count
        }
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frame_folder) 
                         if f.endswith('.png') or f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        return {
            'video_name': video_name,
            'status': 'no_frames',
            'faces_detected': 0,
            'frames_processed': 0
        }
    
    # Process frames
    faces_detected = 0
    frames_processed = 0
    
    def process_frame(frame_file):
        nonlocal faces_detected, frames_processed
        
        frame_path = os.path.join(frame_folder, frame_file)
        frame_num = os.path.splitext(frame_file)[0]
        
        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return
            
            # Detect and align face
            face, confidence = detector.detect_and_align(frame)
            
            if face is not None:
                # Save face
                output_file = os.path.join(output_path, f"{frame_num}.{OUTPUT_FORMAT}")
                
                if OUTPUT_FORMAT == 'jpg':
                    cv2.imwrite(output_file, face, 
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                else:
                    cv2.imwrite(output_file, face)
                
                faces_detected += 1
            
            frames_processed += 1
            
        except Exception as e:
            print(f"Error processing {frame_file}: {e}")
    
    # Process frames with thread pool for I/O
    with ThreadPoolExecutor(max_workers=IO_THREADS) as executor:
        list(tqdm(executor.map(process_frame, frame_files), 
                 total=len(frame_files),
                 desc=f"{video_name}",
                 leave=False))
    
    return {
        'video_name': video_name,
        'status': 'complete',
        'faces_detected': faces_detected,
        'frames_processed': frames_processed,
        'output_path': output_path
    }


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_video_list():
    """Load list of videos from frame mapping"""
    videos = []
    with open(FRAME_MAPPING_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'complete':  # Only process videos with frames
                videos.append(row)
    return videos


def main():
    print("\n" + "=" * 80)
    print("GPU-ACCELERATED FACE DETECTION AND ALIGNMENT")
    print("=" * 80 + "\n")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️ WARNING: No GPU detected! This will be very slow.")
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create output directories
    for output_dir in OUTPUT_DIRS:
        os.makedirs(os.path.join(output_dir, "real FACES"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fake FACES"), exist_ok=True)
    
    # Initialize face detector
    print("\nInitializing RetinaFace detector...")
    detector = FaceDetector(gpu_id=GPU_DEVICE)
    
    # Load video list
    print(f"\nLoading video list from: {FRAME_MAPPING_CSV}")
    videos = load_video_list()
    print(f"✓ Found {len(videos)} videos to process")
    
    # Process videos
    print(f"\nProcessing with {CONCURRENT_VIDEOS} concurrent videos...")
    print(f"Face size: {FACE_SIZE}x{FACE_SIZE}")
    print(f"Min face size: {MIN_FACE_SIZE}px")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Output format: {OUTPUT_FORMAT.upper()}\n")
    
    results = []
    
    from multiprocessing import Pool
    with Pool(processes=CONCURRENT_VIDEOS) as pool:
        # Create tasks
        tasks = [(video, idx, detector) for idx, video in enumerate(videos)]
        
        # Process with progress bar
        for result in tqdm(pool.starmap(process_single_video, tasks),
                          total=len(videos),
                          desc="Overall Progress"):
            results.append(result)
    
    # Save results
    output_json = r"j:\DF\face_detection_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    total_videos = len(results)
    complete = sum(1 for r in results if r['status'] == 'complete')
    total_faces = sum(r['faces_detected'] for r in results)
    total_frames = sum(r['frames_processed'] for r in results)
    
    print(f"\nVideos processed: {complete}/{total_videos}")
    print(f"Total faces detected: {total_faces:,}")
    print(f"Total frames processed: {total_frames:,}")
    print(f"Face detection rate: {total_faces/total_frames*100:.2f}%")
    print(f"\nResults saved to: {output_json}")
    print("\nFaces distributed across drives:")
    for drive in OUTPUT_DIRS:
        print(f"  - {drive}")


if __name__ == "__main__":
    main()
