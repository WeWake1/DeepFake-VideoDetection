"""
Dataset and DataLoader for video sequence sampling.
Loads face sequences from preprocessed data and creates HQ/LQ pairs.
"""

import os
import csv
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

# Handle both direct script execution and package import
try:
    from .utils import apply_lq_transform
except ImportError:
    from utils import apply_lq_transform


class VideoSequenceDataset(Dataset):
    """
    Dataset for loading video face sequences with HQ/LQ streams.
    
    Samples T frames with stride from each video, creates HQ (224x224) and 
    LQ (112x112 + JPEG artifacts) versions for dual-stream training.
    """
    
    def __init__(
        self,
        pairs_csv,
        real_base_path,
        fake_base_path,
        sequence_length=10,
        frame_skip=3,
        hq_size=224,
        lq_size=112,
        lq_jpeg_quality=75,
        augment=True,
        split='train'
    ):
        """
        Args:
            pairs_csv: Path to training_pairs.csv
            real_base_path: Base directory for real face images (e.g., F:/real)
            fake_base_path: Base directory for fake face images (e.g., F:/fake)
            sequence_length: Number of frames to sample (T)
            frame_skip: Stride between frames (matches preprocessing)
            hq_size: HQ stream image size
            lq_size: LQ stream image size
            lq_jpeg_quality: JPEG quality for LQ compression
            augment: Apply data augmentation
            split: 'train' or 'val' (for different augmentation)
        """
        self.real_base = Path(real_base_path)
        self.fake_base = Path(fake_base_path)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.hq_size = hq_size
        self.lq_size = lq_size
        self.lq_jpeg_quality = lq_jpeg_quality
        self.augment = augment and (split == 'train')
        
        # Load video pairs
        self.samples = self._load_pairs(pairs_csv)
        
        # ImageNet normalization for CNNs
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation transforms
        if self.augment:
            self.hq_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.hq_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])
    
    def _load_pairs(self, pairs_csv):
        """Load video pairs from CSV."""
        samples = []
        
        with open(pairs_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both column name formats
                real_video = row.get('real_video', row.get('real_video_name', '')).replace('.mp4', '')
                fake_video = row.get('fake_video', row.get('fake_video_name', '')).replace('.mp4', '')
                
                # Check if directories exist
                real_dir = self.real_base / real_video
                fake_dir = self.fake_base / fake_video
                
                if real_dir.exists() and fake_dir.exists():
                    # Add both real and fake as separate samples
                    samples.append({
                        'video_dir': real_dir,
                        'label': 0,  # real
                        'video_name': real_video
                    })
                    samples.append({
                        'video_dir': fake_dir,
                        'label': 1,  # fake
                        'video_name': fake_video
                    })
        
        print(f"Loaded {len(samples)} video samples ({len(samples)//2} pairs)")
        return samples
    
    def _get_frame_paths(self, video_dir):
        """Get sorted list of face image paths from video directory."""
        frames = sorted(list(video_dir.glob('*.jpg')))
        return frames
    
    def _sample_sequence(self, frame_paths):
        """
        Sample a sequence of T frames with stride.
        
        Args:
            frame_paths: List of all available frame paths
        
        Returns:
            list: Selected frame paths (length T)
        """
        num_frames = len(frame_paths)
        
        # Calculate required span for sequence
        required_span = (self.sequence_length - 1) * self.frame_skip + 1
        
        if num_frames < required_span:
            # Video too short - sample uniformly and pad if needed
            if num_frames == 0:
                return None
            
            # Sample what we can with wrapping
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            return [frame_paths[i] for i in indices]
        
        # Random start index for augmentation (train) or fixed (val)
        if self.augment:
            max_start = num_frames - required_span
            start_idx = random.randint(0, max_start)
        else:
            # Use middle of video for validation
            max_start = num_frames - required_span
            start_idx = max_start // 2
        
        # Sample with stride
        indices = range(start_idx, start_idx + required_span, self.frame_skip)
        sampled = [frame_paths[i] for i in indices]
        
        return sampled[:self.sequence_length]  # Ensure exactly T frames
    
    def _load_image(self, path):
        """Load image as RGB numpy array."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _apply_temporal_augmentation(self, sequence):
        """
        Apply temporal augmentation (random reverse).
        
        Args:
            sequence: List of images
        
        Returns:
            list: Augmented sequence
        """
        if self.augment and random.random() < 0.3:
            return sequence[::-1]  # Reverse temporal order
        return sequence
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            hq_sequence: Tensor (T, 3, 224, 224)
            lq_sequence: Tensor (T, 3, 112, 112)
            label: 0 (real) or 1 (fake)
        """
        sample = self.samples[idx]
        video_dir = sample['video_dir']
        label = sample['label']
        
        # Get frame paths
        frame_paths = self._get_frame_paths(video_dir)
        
        # Sample sequence
        sampled_paths = self._sample_sequence(frame_paths)
        
        if sampled_paths is None or len(sampled_paths) == 0:
            # Fallback: return random other sample
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Load images
        hq_images = [self._load_image(p) for p in sampled_paths]
        
        # Temporal augmentation
        hq_images = self._apply_temporal_augmentation(hq_images)
        
        # Create HQ tensors
        hq_tensors = []
        for img in hq_images:
            # Resize to HQ size
            img_resized = cv2.resize(img, (self.hq_size, self.hq_size), interpolation=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(img_resized)
            
            # Apply transforms (includes normalization)
            tensor = self.hq_transform(img_pil)
            hq_tensors.append(tensor)
        
        hq_sequence = torch.stack(hq_tensors)  # (T, 3, 224, 224)
        
        # Create LQ tensors
        lq_tensors = []
        for img in hq_images:
            # Apply LQ transform (downsample + JPEG)
            lq_img = apply_lq_transform(img, self.lq_size, self.lq_jpeg_quality)
            lq_pil = Image.fromarray(lq_img)
            
            # Same augmentation state as HQ (color jitter handled via same random state)
            # Just normalize
            lq_tensor = transforms.ToTensor()(lq_pil)
            lq_tensor = self.normalize(lq_tensor)
            lq_tensors.append(lq_tensor)
        
        lq_sequence = torch.stack(lq_tensors)  # (T, 3, 112, 112)
        
        return hq_sequence, lq_sequence, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    pairs_csv,
    real_base_path,
    fake_base_path,
    batch_size=12,
    num_workers=4,
    train_split=0.85,
    sequence_length=10,
    frame_skip=3,
    seed=42
):
    """
    Create train and validation DataLoaders with stratified split.
    
    Args:
        pairs_csv: Path to training_pairs.csv
        real_base_path: Real faces base directory
        fake_base_path: Fake faces base directory
        batch_size: Batch size
        num_workers: DataLoader workers
        train_split: Fraction for training (rest for validation)
        sequence_length: Frames per sequence
        frame_skip: Frame stride
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset
    full_dataset = VideoSequenceDataset(
        pairs_csv=pairs_csv,
        real_base_path=real_base_path,
        fake_base_path=fake_base_path,
        sequence_length=sequence_length,
        frame_skip=frame_skip,
        augment=True,
        split='train'
    )
    
    # Stratified split by label
    real_indices = [i for i, s in enumerate(full_dataset.samples) if s['label'] == 0]
    fake_indices = [i for i, s in enumerate(full_dataset.samples) if s['label'] == 1]
    
    random.shuffle(real_indices)
    random.shuffle(fake_indices)
    
    n_train_real = int(len(real_indices) * train_split)
    n_train_fake = int(len(fake_indices) * train_split)
    
    train_indices = real_indices[:n_train_real] + fake_indices[:n_train_fake]
    val_indices = real_indices[n_train_real:] + fake_indices[n_train_fake:]
    
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
    
    # Create train dataset
    train_dataset = VideoSequenceDataset(
        pairs_csv=pairs_csv,
        real_base_path=real_base_path,
        fake_base_path=fake_base_path,
        sequence_length=sequence_length,
        frame_skip=frame_skip,
        augment=True,
        split='train'
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    
    # Create val dataset
    val_dataset = VideoSequenceDataset(
        pairs_csv=pairs_csv,
        real_base_path=real_base_path,
        fake_base_path=fake_base_path,
        sequence_length=sequence_length,
        frame_skip=frame_skip,
        augment=False,
        split='val'
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader
