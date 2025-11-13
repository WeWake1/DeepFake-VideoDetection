"""
Run inference on DFD (Google/Jigsaw) dataset
Evaluates trained model on DFD test videos
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'train'))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from inference import VideoInference
from models import create_model
from utils import load_checkpoint

# ===== CONFIGURATION =====
MODEL_PATH = r"j:\DF\checkpoints\model_alpha_celeb_only.pth"
CONFIG_PATH = r"j:\DF\config\defaults.yaml"
TEST_CSV = r"j:\DF\evaluation\dfd\dfd_test_videos_ready.csv"
OUTPUT_CSV = r"j:\DF\evaluation\dfd\dfd_results.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1

def load_model_and_config():
    """Load trained model and configuration"""
    print("="*70)
    print("🔧 LOADING MODEL")
    print("="*70)
    
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded from: {CONFIG_PATH}")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded from: {MODEL_PATH}")
    print(f"✓ Device: {DEVICE}")
    
    return model, config

def run_inference_on_dfd():
    """Run inference on DFD test videos"""
    print("="*70)
    print("🎯 DFD INFERENCE")
    print("="*70)
    print(f"   Model: {MODEL_PATH}")
    print(f"   Test CSV: {TEST_CSV}")
    print(f"   Output: {OUTPUT_CSV}")
    print("="*70)
    
    # Load model
    model, config = load_model_and_config()
    
    # Create inference wrapper
    inferencer = VideoInference(model, DEVICE, config)
    
    # Load test videos
    test_df = pd.read_csv(TEST_CSV)
    print(f"\n✓ Loaded {len(test_df)} test videos")
    print(f"   Real: {(test_df['label'] == 'real').sum()}")
    print(f"   Fake: {(test_df['label'] == 'fake').sum()}")
    
    # Run inference
    results = []
    
    print("\n🔍 Running inference...")
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
            video_name = row['video_name']
            true_label = row['label']
            folder = row['folder']
            faces_path = row['faces_path']
            
            try:
                # Run inference
                pred_dict = inferencer.predict_video(faces_path)
                
                # Extract results
                confidence = pred_dict['score']
                prediction = 'fake' if confidence > 0.5 else 'real'
                num_frames = pred_dict['num_frames']
                
                results.append({
                    'video_name': video_name,
                    'true_label': true_label,
                    'folder': folder,
                    'predicted_label': prediction,
                    'confidence': confidence,
                    'num_frames': num_frames,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"\n⚠️  Error processing {video_name}: {e}")
                results.append({
                    'video_name': video_name,
                    'true_label': true_label,
                    'folder': folder,
                    'predicted_label': 'error',
                    'confidence': 0.5,
                    'num_frames': 0,
                    'status': f'error: {str(e)}'
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Results saved to: {OUTPUT_CSV}")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📊 DFD TEST RESULTS")
    print("="*70)
    
    successful = results_df[results_df['status'] == 'success']
    
    if len(successful) > 0:
        # Overall accuracy
        correct = (successful['true_label'] == successful['predicted_label']).sum()
        accuracy = correct / len(successful) * 100
        
        # Real accuracy
        real_vids = successful[successful['true_label'] == 'real']
        real_correct = (real_vids['true_label'] == real_vids['predicted_label']).sum()
        real_acc = real_correct / len(real_vids) * 100 if len(real_vids) > 0 else 0
        
        # Fake accuracy
        fake_vids = successful[successful['true_label'] == 'fake']
        fake_correct = (fake_vids['true_label'] == fake_vids['predicted_label']).sum()
        fake_acc = fake_correct / len(fake_vids) * 100 if len(fake_vids) > 0 else 0
        
        print(f"   Total Videos: {len(successful)}")
        print(f"   Correct: {correct}/{len(successful)}")
        print(f"   Overall Accuracy: {accuracy:.2f}%")
        print(f"   ")
        print(f"   Real Videos: {len(real_vids)}")
        print(f"   Real Correct: {real_correct}/{len(real_vids)}")
        print(f"   Real Accuracy: {real_acc:.2f}%")
        print(f"   ")
        print(f"   Fake Videos: {len(fake_vids)}")
        print(f"   Fake Correct: {fake_correct}/{len(fake_vids)}")
        print(f"   Fake Accuracy: {fake_acc:.2f}%")
        
        # Confidence statistics
        print(f"\n   Confidence Statistics:")
        print(f"   Mean: {successful['confidence'].mean():.4f}")
        print(f"   Std: {successful['confidence'].std():.4f}")
        print(f"   Min: {successful['confidence'].min():.4f}")
        print(f"   Max: {successful['confidence'].max():.4f}")
        
        # Error analysis
        errors = successful[successful['true_label'] != successful['predicted_label']]
        if len(errors) > 0:
            print(f"\n   ⚠️  Misclassified videos ({len(errors)}):")
            for _, err in errors.head(10).iterrows():
                print(f"      - {err['video_name']}: true={err['true_label']}, pred={err['predicted_label']}, conf={err['confidence']:.4f}")
        
    print("="*70)
    
    return results_df

if __name__ == "__main__":
    run_inference_on_dfd()
