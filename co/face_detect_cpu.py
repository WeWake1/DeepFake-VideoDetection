"""
CPU-Optimized Face Detection, Cropping, and Alignment
Uses RetinaFace with maximum CPU parallelization
Processes frames from extracted videos using all CPU cores
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
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

# CPU settings
NUM_WORKERS = max(1, cpu_count() - 8)  # Leave 8 cores for system
FRAMES_PER_WORKER = 16  # Frames to process per worker batch
IO_THREADS = 64  # Threads for file I/O

# Resume capability
RESUME = True  # Skip already processed videos

# Output format
OUTPUT_FORMAT = "jpg"  # jpg or png
JPEG_QUALITY = 95  # JPEG quality (1-100)

# ============================================================================
# FACE DETECTION AND ALIGNMENT
# ============================================================================

def align_face(img, facial_area, left_eye, right_eye):
    """
    Align face based on eye positions (essential for deepfake detection)
    """
    try:
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
    except Exception as e:
        return None


def detect_and_align_face(frame):
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
        aligned_face = align_face(frame, facial_area, left_eye, right_eye)
        
        if aligned_face is None:
            return None, 0.0
        
        # Resize to standard size
        face_resized = cv2.resize(aligned_face, (FACE_SIZE, FACE_SIZE), 
                                 interpolation=cv2.INTER_LANCZOS4)
        
        return face_resized, best_conf
        
    except Exception as e:
        return None, 0.0


# ============================================================================
# FRAME PROCESSING
# ============================================================================

def process_frame_batch(args):
    """
    Process a batch of frames (called by worker process)
    """
    frame_paths, output_dir = args
    
    results = []
    for frame_path in frame_paths:
        frame_file = os.path.basename(frame_path)
        frame_num = os.path.splitext(frame_file)[0]
        
        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                results.append((frame_num, False, 0.0))
                continue
            
            # Detect and align face
            face, confidence = detect_and_align_face(frame)
            
            if face is not None:
                # Save face
                output_file = os.path.join(output_dir, f"{frame_num}.{OUTPUT_FORMAT}")
                
                if OUTPUT_FORMAT == 'jpg':
                    cv2.imwrite(output_file, face, 
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                else:
                    cv2.imwrite(output_file, face)
                
                results.append((frame_num, True, confidence))
            else:
                results.append((frame_num, False, 0.0))
                
        except Exception as e:
            results.append((frame_num, False, 0.0))
    
    return results


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


def process_single_video(args):
    """
    Process one video: detect faces in all frames using CPU parallelization
    """
    video_info, video_idx = args
    
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
    frame_files = sorted([os.path.join(frame_folder, f) 
                         for f in os.listdir(frame_folder) 
                         if f.endswith('.png') or f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        return {
            'video_name': video_name,
            'status': 'no_frames',
            'faces_detected': 0,
            'frames_processed': 0
        }
    
    # Split frames into batches for parallel processing
    batches = []
    for i in range(0, len(frame_files), FRAMES_PER_WORKER):
        batch = frame_files[i:i + FRAMES_PER_WORKER]
        batches.append((batch, output_path))
    
    # Process batches in parallel
    faces_detected = 0
    frames_processed = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_results = list(tqdm(executor.map(process_frame_batch, batches),
                                 total=len(batches),
                                 desc=f"{video_name}",
                                 leave=False))
        
        for batch_result in batch_results:
            for frame_num, success, conf in batch_result:
                frames_processed += 1
                if success:
                    faces_detected += 1
    
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
    print("CPU-OPTIMIZED FACE DETECTION AND ALIGNMENT")
    print("=" * 80 + "\n")
    
    # Show CPU info
    total_cores = cpu_count()
    print(f"CPU cores available: {total_cores}")
    print(f"Worker processes: {NUM_WORKERS}")
    print(f"I/O threads: {IO_THREADS}")
    
    # Create output directories
    for output_dir in OUTPUT_DIRS:
        os.makedirs(os.path.join(output_dir, "real FACES"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fake FACES"), exist_ok=True)
    
    # Load video list
    print(f"\nLoading video list from: {FRAME_MAPPING_CSV}")
    videos = load_video_list()
    print(f"✓ Found {len(videos)} videos to process")
    
    # Process videos
    print(f"\nProcessing videos...")
    print(f"Face size: {FACE_SIZE}x{FACE_SIZE}")
    print(f"Min face size: {MIN_FACE_SIZE}px")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Output format: {OUTPUT_FORMAT.upper()}\n")
    
    results = []
    
    # Process videos sequentially (each uses all CPU cores internally)
    for idx, video in enumerate(tqdm(videos, desc="Overall Progress")):
        result = process_single_video((video, idx))
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
    if total_frames > 0:
        print(f"Face detection rate: {total_faces/total_frames*100:.2f}%")
    print(f"\nResults saved to: {output_json}")
    print("\nFaces distributed across drives:")
    for drive in OUTPUT_DIRS:
        print(f"  - {drive}")


if __name__ == "__main__":
    main()
