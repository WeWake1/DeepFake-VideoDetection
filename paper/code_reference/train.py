"""
Training script for deepfake detection model.
Handles full training loop with AMP, checkpointing, and early stopping.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Add train directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import create_dataloaders
from models import create_model
from utils import (
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    EarlyStopping,
    count_parameters
)


def train_epoch(model, dataloader, optimizer, scaler, criterion, device, gradient_clip=1.0, use_amp=True):
    """
    Train for one epoch.
    
    Args:
        model: DeepfakeDetector model
        dataloader: Training DataLoader
        optimizer: Optimizer
        scaler: GradScaler for AMP
        criterion: Loss function
        device: torch device
        gradient_clip: Gradient clipping threshold
        use_amp: Use automatic mixed precision
    
    Returns:
        dict: Training metrics
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (hq_seq, lq_seq, labels) in enumerate(pbar):
        # Move to device
        hq_seq = hq_seq.to(device)
        lq_seq = lq_seq.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B,) -> (B, 1)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if use_amp:
            with autocast():
                logits = model(hq_seq, lq_seq)
                loss = criterion(logits, labels)
        else:
            logits = model(hq_seq, lq_seq)
            loss = criterion(logits, labels)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        with torch.no_grad():
            scores = torch.sigmoid(logits).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_scores.extend(scores.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    metrics['loss'] = avg_loss
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, use_amp=True):
    """
    Validate for one epoch.
    
    Args:
        model: DeepfakeDetector model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: torch device
        use_amp: Use automatic mixed precision
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for hq_seq, lq_seq, labels in pbar:
            # Move to device
            hq_seq = hq_seq.to(device)
            lq_seq = lq_seq.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward pass
            if use_amp:
                with autocast():
                    logits = model(hq_seq, lq_seq)
                    loss = criterion(logits, labels)
            else:
                logits = model(hq_seq, lq_seq)
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            # Predictions
            scores = torch.sigmoid(logits).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_scores.extend(scores.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    metrics['loss'] = avg_loss
    
    return metrics


def main(config_path):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Deepfake Detection Training")
    print("="*60)
    
    # Set random seeds
    seed = config['hardware']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # Verify GPU will be used
        test_tensor = torch.randn(1, 1).to(device)
        assert test_tensor.is_cuda, "ERROR: Tensor not on GPU!"
        print("✓ GPU verification successful - tensors will use CUDA")
    else:
        print("\n⚠ WARNING: Training on CPU - this will be EXTREMELY slow!")
        print("Expected time: 10-20 days instead of 1-2 days")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting. Please ensure CUDA is properly installed.")
            sys.exit(1)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        pairs_csv=config['data']['training_pairs'],
        real_base_path=config['data']['real_path'],
        fake_base_path=config['data']['fake_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        train_split=config['data']['train_split'],
        sequence_length=config['data']['sequence_length'],
        frame_skip=config['data']['frame_skip'],
        seed=seed
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nBuilding model...")
    model = create_model(config['model']).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['optimization']['scheduler_factor'],
        patience=config['optimization']['scheduler_patience'],
        min_lr=config['optimization']['scheduler_min_lr'],
        verbose=True
    )
    
    # AMP scaler
    use_amp = config['optimization']['use_amp'] and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stop_patience'],
        mode='min'
    )
    
    # Setup logging
    logger = setup_logging(config['paths']['logs'])
    
    # Create checkpoint directory
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            gradient_clip=config['training']['gradient_clip'],
            use_amp=use_amp
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device,
            use_amp=use_amp
        )
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc_roc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc_roc']:.4f}")
        
        # Log metrics
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'train_auc': train_metrics['auc_roc'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_auc': val_metrics['auc_roc'],
            'lr': optimizer.param_groups[0]['lr']
        }
        logger.log(log_dict)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                val_metrics, config,
                config['paths']['best_model']
            )
            print(f"✓ Best model saved (val_loss: {best_val_loss:.4f})")
        
        # Save last model
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            val_metrics, config,
            config['paths']['last_model']
        )
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Close logger
    logger.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved: {config['paths']['best_model']}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/defaults.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    main(args.config)
