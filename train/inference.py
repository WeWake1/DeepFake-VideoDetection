"""
Inference and evaluation script for trained deepfake detection model.
"""

import os
import sys
import argparse
import yaml
import csv
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# Add train directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from utils import (
    calculate_metrics,
    load_checkpoint,
    save_predictions,
    apply_lq_transform
)
from torchvision import transforms


class VideoInference:
    """
    Inference wrapper for scoring individual videos.
    """
    
    def __init__(self, model, device, config):
        """
        Args:
            model: Trained DeepfakeDetector
            device: torch device
            config: Configuration dict
        """
        self.model = model
        self.device = device
        self.config = config
        
        self.model.eval()
        
        # Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.hq_size = config.get('preprocessing', {}).get('hq_size', 224)
        self.lq_size = config.get('preprocessing', {}).get('lq_size', 112)
        self.sequence_length = config.get('data', {}).get('sequence_length', 10)
        self.frame_skip = config.get('data', {}).get('frame_skip', 3)
    
    def load_video_frames(self, video_dir):
        """
        Load face images from video directory.
        
        Args:
            video_dir: Path to directory with face images
        
        Returns:
            list: Sorted list of image paths
        """
        video_path = Path(video_dir)
        frames = sorted(list(video_path.glob('*.jpg')))
        return frames
    
    def sample_frames(self, frame_paths):
        """
        Sample T frames with stride from video.
        
        Args:
            frame_paths: List of all frame paths
        
        Returns:
            list: Sampled frame paths
        """
        num_frames = len(frame_paths)
        
        if num_frames == 0:
            return None
        
        # Calculate required span
        required_span = (self.sequence_length - 1) * self.frame_skip + 1
        
        if num_frames < required_span:
            # Too short - sample uniformly
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            return [frame_paths[i] for i in indices]
        
        # Use middle of video
        max_start = num_frames - required_span
        start_idx = max_start // 2
        
        # Sample with stride
        indices = range(start_idx, start_idx + required_span, self.frame_skip)
        sampled = [frame_paths[i] for i in indices]
        
        return sampled[:self.sequence_length]
    
    def preprocess_image(self, img_path):
        """
        Load and preprocess a single image for both HQ and LQ streams.
        
        Args:
            img_path: Path to face image
        
        Returns:
            tuple: (hq_tensor, lq_tensor)
        """
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # HQ stream
        hq = cv2.resize(img_np, (self.hq_size, self.hq_size), interpolation=cv2.INTER_LINEAR)
        hq_tensor = transforms.ToTensor()(hq)
        hq_tensor = self.normalize(hq_tensor)
        
        # LQ stream
        lq = apply_lq_transform(img_np, self.lq_size, 75)
        lq_tensor = transforms.ToTensor()(lq)
        lq_tensor = self.normalize(lq_tensor)
        
        return hq_tensor, lq_tensor
    
    def predict_video(self, video_dir):
        """
        Predict score for a single video.
        
        Args:
            video_dir: Path to video face directory
        
        Returns:
            dict: Prediction result with score and binary prediction
        """
        # Load frames
        frame_paths = self.load_video_frames(video_dir)
        
        if not frame_paths:
            return {'score': 0.5, 'prediction': 0, 'error': 'No frames found'}
        
        # Sample sequence
        sampled_paths = self.sample_frames(frame_paths)
        
        if sampled_paths is None:
            return {'score': 0.5, 'prediction': 0, 'error': 'Insufficient frames'}
        
        # Preprocess
        hq_tensors = []
        lq_tensors = []
        
        for path in sampled_paths:
            hq_t, lq_t = self.preprocess_image(path)
            hq_tensors.append(hq_t)
            lq_tensors.append(lq_t)
        
        # Stack into sequences
        hq_seq = torch.stack(hq_tensors).unsqueeze(0)  # (1, T, 3, H, W)
        lq_seq = torch.stack(lq_tensors).unsqueeze(0)
        
        # Move to device
        hq_seq = hq_seq.to(self.device)
        lq_seq = lq_seq.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(hq_seq, lq_seq)
            score = torch.sigmoid(logits).item()
            prediction = 1 if score > 0.5 else 0
        
        return {
            'score': score,
            'prediction': prediction,
            'num_frames': len(sampled_paths)
        }


def evaluate_test_set(model, device, config, test_pairs_csv, output_dir='results'):
    """
    Evaluate model on test set and save results.
    
    Args:
        model: Trained model
        device: torch device
        config: Configuration dict
        test_pairs_csv: Path to test pairs CSV (or use validation split)
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference
    inference = VideoInference(model, device, config)
    
    # Load test samples
    print("Loading test samples...")
    test_samples = []
    
    # Read from pairs CSV
    real_base = Path(config['data']['real_path'])
    fake_base = Path(config['data']['fake_path'])
    
    with open(test_pairs_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both column name formats
            real_video = row.get('real_video', row.get('real_video_name', '')).replace('.mp4', '')
            fake_video = row.get('fake_video', row.get('fake_video_name', '')).replace('.mp4', '')
            
            real_dir = real_base / real_video
            fake_dir = fake_base / fake_video
            
            if real_dir.exists():
                test_samples.append({
                    'video_dir': str(real_dir),
                    'video_name': real_video,
                    'label': 0
                })
            
            if fake_dir.exists():
                test_samples.append({
                    'video_dir': str(fake_dir),
                    'video_name': fake_video,
                    'label': 1
                })
    
    # Take subset for validation (last 15%)
    split_idx = int(len(test_samples) * 0.85)
    test_samples = test_samples[split_idx:]
    
    print(f"Testing on {len(test_samples)} videos...")
    
    # Run predictions
    predictions = []
    all_labels = []
    all_scores = []
    all_preds = []
    
    for sample in tqdm(test_samples, desc='Evaluating'):
        result = inference.predict_video(sample['video_dir'])
        
        predictions.append({
            'video_name': sample['video_name'],
            'label': sample['label'],
            'prediction': result['prediction'],
            'score': result['score']
        })
        
        if 'error' not in result:
            all_labels.append(sample['label'])
            all_scores.append(result['score'])
            all_preds.append(result['prediction'])
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    
    # Print results
    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("="*60)
    
    if 'true_positives' in metrics:
        print("\nConfusion Matrix:")
        print(f"TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
        print(f"FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")
    
    # Save results
    save_predictions(predictions, os.path.join(output_dir, 'predictions.csv'))
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/defaults.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test-pairs',
        type=str,
        default='training_pairs.csv',
        help='Path to test pairs CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        help='Score a single video directory'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(config['model']).to(device)
    load_checkpoint(args.checkpoint, model)
    
    # Single video or full test set
    if args.video_dir:
        # Score single video
        inference = VideoInference(model, device, config)
        result = inference.predict_video(args.video_dir)
        
        print("\n" + "="*60)
        print(f"Video: {args.video_dir}")
        print(f"Score: {result['score']:.4f}")
        print(f"Prediction: {'FAKE' if result['prediction'] == 1 else 'REAL'}")
        print("="*60)
    else:
        # Evaluate full test set
        evaluate_test_set(model, device, config, args.test_pairs, args.output_dir)


if __name__ == '__main__':
    main()
