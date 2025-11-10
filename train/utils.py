"""
Utility functions for training, evaluation, and checkpointing.
"""

import os
import json
import csv
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import cv2


def calculate_metrics(y_true, y_pred, y_scores):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_pred: Binary predictions (0 or 1)
        y_scores: Prediction probabilities [0,1]
    
    Returns:
        dict: All computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, path):
    """
    Save model checkpoint with full training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        metrics: Dict of validation metrics
        config: Training configuration dict
        path: Checkpoint save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint and optionally restore optimizer/scheduler state.
    
    Args:
        path: Checkpoint file path
        model: Model instance to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
    
    Returns:
        dict: Checkpoint metadata (epoch, metrics, config)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }


class CSVLogger:
    """
    Simple CSV logger for epoch-level metrics.
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.fieldnames = None
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Initialize file
        self.file = open(log_path, 'w', newline='')
        self.writer = None
    
    def log(self, metrics_dict):
        """Log a row of metrics."""
        if self.writer is None:
            self.fieldnames = list(metrics_dict.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        
        self.writer.writerow(metrics_dict)
        self.file.flush()
    
    def close(self):
        """Close the log file."""
        if self.file:
            self.file.close()


def apply_lq_transform(image, target_size=112, jpeg_quality=75):
    """
    Apply low-quality transformation to create LQ stream input.
    Downsamples and adds JPEG compression artifacts.
    
    Args:
        image: numpy array (H, W, 3) in RGB, uint8
        target_size: Target resolution (default 112x112)
        jpeg_quality: JPEG compression quality (default 75)
    
    Returns:
        numpy array: Transformed LQ image (target_size, target_size, 3)
    """
    # Resize
    lq = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Apply JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(lq, cv2.COLOR_RGB2BGR), encode_param)
    lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
    
    return lq


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, current_value):
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
        
        Returns:
            bool: True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False


def setup_logging(log_dir):
    """
    Setup logging directory and return logger instance.
    
    Args:
        log_dir: Directory for log files
    
    Returns:
        CSVLogger: Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'training_log.csv')
    return CSVLogger(log_path)


def save_predictions(predictions, output_path):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: List of dicts with keys: video_name, label, prediction, score
        output_path: Output CSV path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['video_name', 'true_label', 'prediction', 'score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in predictions:
            writer.writerow({
                'video_name': pred['video_name'],
                'true_label': pred['label'],
                'prediction': pred['prediction'],
                'score': pred['score']
            })
    
    print(f"✓ Predictions saved: {output_path}")


def count_parameters(model):
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
