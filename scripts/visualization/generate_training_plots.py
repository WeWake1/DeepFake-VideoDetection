"""
Generate publication-quality training curve plots.

Creates:
1. Loss curves (train + val)
2. Accuracy curves (train + val)
3. AUC curves (train + val)
4. Combined multi-panel figure

Saves to paper/figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load data
log_file = Path(r'J:\DF\logs\training_log.csv')
output_dir = Path(r'J:\DF\paper\figures')
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(log_file)

print("=" * 70)
print("GENERATING TRAINING PLOTS")
print("=" * 70)
print(f"\nLoaded {len(df)} epochs from {log_file}")
print(f"Output directory: {output_dir}\n")

# Create individual plots
fig_size = (10, 6)
colors = {'train': '#2E86AB', 'val': '#A23B72'}

# ============================================================================
# 1. LOSS CURVES
# ============================================================================
print("📊 Creating loss curves...")
fig, ax = plt.subplots(figsize=fig_size)

ax.plot(df['epoch'], df['train_loss'], 
        marker='o', linewidth=2, markersize=5, 
        label='Training Loss', color=colors['train'], alpha=0.8)
ax.plot(df['epoch'], df['val_loss'], 
        marker='s', linewidth=2, markersize=5, 
        label='Validation Loss', color=colors['val'], alpha=0.8)

ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Binary Cross-Entropy Loss', fontweight='bold')
ax.set_title('Training and Validation Loss Curves', fontweight='bold', pad=15)
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')  # Log scale for better visualization
ax.set_xlim(0.5, len(df) + 0.5)

# Add annotations for key points
min_val_loss_idx = df['val_loss'].idxmin()
ax.annotate(f'Best: Epoch {df.loc[min_val_loss_idx, "epoch"]}\nVal Loss: {df.loc[min_val_loss_idx, "val_loss"]:.6f}',
            xy=(df.loc[min_val_loss_idx, 'epoch'], df.loc[min_val_loss_idx, 'val_loss']),
            xytext=(10, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))

plt.tight_layout()
loss_file = output_dir / 'training_loss.png'
plt.savefig(loss_file, bbox_inches='tight')
print(f"   ✓ Saved: {loss_file}")
plt.close()

# ============================================================================
# 2. ACCURACY CURVES
# ============================================================================
print("📊 Creating accuracy curves...")
fig, ax = plt.subplots(figsize=fig_size)

ax.plot(df['epoch'], df['train_acc'] * 100, 
        marker='o', linewidth=2, markersize=5, 
        label='Training Accuracy', color=colors['train'], alpha=0.8)
ax.plot(df['epoch'], df['val_acc'] * 100, 
        marker='s', linewidth=2, markersize=5, 
        label='Validation Accuracy', color=colors['val'], alpha=0.8)

ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Training and Validation Accuracy', fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.5, len(df) + 0.5)
ax.set_ylim(90, 100.5)

# Add 100% accuracy line
ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='100% Accuracy')

# Add annotation for 100% val accuracy achievement
val_100_epochs = df[df['val_acc'] >= 0.9999]['epoch'].tolist()
if val_100_epochs:
    first_100 = val_100_epochs[0]
    ax.annotate(f'100% Val Accuracy\nAchieved at Epoch {first_100}',
                xy=(first_100, 100),
                xytext=(15, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))

plt.tight_layout()
acc_file = output_dir / 'training_accuracy.png'
plt.savefig(acc_file, bbox_inches='tight')
print(f"   ✓ Saved: {acc_file}")
plt.close()

# ============================================================================
# 3. AUC CURVES
# ============================================================================
print("📊 Creating AUC curves...")
fig, ax = plt.subplots(figsize=fig_size)

ax.plot(df['epoch'], df['train_auc'], 
        marker='o', linewidth=2, markersize=5, 
        label='Training AUC', color=colors['train'], alpha=0.8)
ax.plot(df['epoch'], df['val_auc'], 
        marker='s', linewidth=2, markersize=5, 
        label='Validation AUC', color=colors['val'], alpha=0.8)

ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('AUC-ROC Score', fontweight='bold')
ax.set_title('Training and Validation AUC-ROC', fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.5, len(df) + 0.5)
ax.set_ylim(0.97, 1.001)

# Add perfect AUC line
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect AUC')

# Add annotation for AUC=1.0 achievement
val_auc_1_epochs = df[df['val_auc'] >= 0.9999]['epoch'].tolist()
if val_auc_1_epochs:
    first_auc_1 = val_auc_1_epochs[0]
    ax.annotate(f'AUC ≈ 1.0\nAchieved at Epoch {first_auc_1}',
                xy=(first_auc_1, df.loc[df['epoch'] == first_auc_1, 'val_auc'].values[0]),
                xytext=(15, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))

plt.tight_layout()
auc_file = output_dir / 'training_auc.png'
plt.savefig(auc_file, bbox_inches='tight')
print(f"   ✓ Saved: {auc_file}")
plt.close()

# ============================================================================
# 4. LEARNING RATE SCHEDULE
# ============================================================================
print("📊 Creating learning rate schedule...")
fig, ax = plt.subplots(figsize=fig_size)

ax.plot(df['epoch'], df['lr'], 
        marker='o', linewidth=2, markersize=5, 
        color='#F77F00', alpha=0.8)

ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Learning Rate', fontweight='bold')
ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)', fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')
ax.set_xlim(0.5, len(df) + 0.5)

# Add annotations for LR changes
lr_changes = df[df['lr'] != df['lr'].shift()].index[1:]  # Skip first epoch
for idx in lr_changes:
    ax.axvline(x=df.loc[idx, 'epoch'], color='red', linestyle=':', alpha=0.5)
    ax.text(df.loc[idx, 'epoch'], df.loc[idx, 'lr'], 
            f"  LR={df.loc[idx, 'lr']:.0e}", 
            verticalalignment='bottom', fontsize=9)

plt.tight_layout()
lr_file = output_dir / 'learning_rate.png'
plt.savefig(lr_file, bbox_inches='tight')
print(f"   ✓ Saved: {lr_file}")
plt.close()

# ============================================================================
# 5. COMBINED MULTI-PANEL FIGURE (for paper)
# ============================================================================
print("📊 Creating combined multi-panel figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dual-Stream EfficientNet Training Dynamics on Celeb-DF', 
             fontsize=16, fontweight='bold', y=0.995)

# Panel A: Loss
ax = axes[0, 0]
ax.plot(df['epoch'], df['train_loss'], marker='o', linewidth=2, markersize=4, 
        label='Train', color=colors['train'], alpha=0.8)
ax.plot(df['epoch'], df['val_loss'], marker='s', linewidth=2, markersize=4, 
        label='Validation', color=colors['val'], alpha=0.8)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('BCE Loss (log scale)', fontweight='bold')
ax.set_title('(A) Loss Curves', fontweight='bold', loc='left')
ax.legend(loc='best', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')
ax.set_xlim(0.5, len(df) + 0.5)

# Panel B: Accuracy
ax = axes[0, 1]
ax.plot(df['epoch'], df['train_acc'] * 100, marker='o', linewidth=2, markersize=4, 
        label='Train', color=colors['train'], alpha=0.8)
ax.plot(df['epoch'], df['val_acc'] * 100, marker='s', linewidth=2, markersize=4, 
        label='Validation', color=colors['val'], alpha=0.8)
ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('(B) Accuracy Curves', fontweight='bold', loc='left')
ax.legend(loc='lower right', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.5, len(df) + 0.5)
ax.set_ylim(90, 100.5)

# Panel C: AUC
ax = axes[1, 0]
ax.plot(df['epoch'], df['train_auc'], marker='o', linewidth=2, markersize=4, 
        label='Train', color=colors['train'], alpha=0.8)
ax.plot(df['epoch'], df['val_auc'], marker='s', linewidth=2, markersize=4, 
        label='Validation', color=colors['val'], alpha=0.8)
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('AUC-ROC', fontweight='bold')
ax.set_title('(C) AUC-ROC Curves', fontweight='bold', loc='left')
ax.legend(loc='lower right', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.5, len(df) + 0.5)
ax.set_ylim(0.97, 1.001)

# Panel D: Learning Rate
ax = axes[1, 1]
ax.plot(df['epoch'], df['lr'], marker='o', linewidth=2, markersize=4, 
        color='#F77F00', alpha=0.8)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Learning Rate (log scale)', fontweight='bold')
ax.set_title('(D) Learning Rate Schedule', fontweight='bold', loc='left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_yscale('log')
ax.set_xlim(0.5, len(df) + 0.5)

# Highlight LR reduction points
for idx in lr_changes:
    ax.axvline(x=df.loc[idx, 'epoch'], color='red', linestyle=':', alpha=0.3)

plt.tight_layout()
combined_file = output_dir / 'training_curves_combined.png'
plt.savefig(combined_file, bbox_inches='tight')
print(f"   ✓ Saved: {combined_file}")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY STATISTICS")
print("=" * 70)
print(f"\nTotal epochs: {len(df)}")
print(f"Initial learning rate: {df['lr'].iloc[0]:.0e}")
print(f"Final learning rate: {df['lr'].iloc[-1]:.0e}")
print(f"\nBest validation results:")
print(f"  - Min loss: {df['val_loss'].min():.6f} (Epoch {df['val_loss'].idxmin() + 1})")
print(f"  - Max accuracy: {df['val_acc'].max() * 100:.2f}% (Epoch {df['val_acc'].idxmax() + 1})")
print(f"  - Max AUC: {df['val_auc'].max():.6f} (Epoch {df['val_auc'].idxmax() + 1})")

print(f"\nFinal epoch results:")
print(f"  - Train loss: {df['train_loss'].iloc[-1]:.6f}")
print(f"  - Train acc: {df['train_acc'].iloc[-1] * 100:.2f}%")
print(f"  - Val loss: {df['val_loss'].iloc[-1]:.6f}")
print(f"  - Val acc: {df['val_acc'].iloc[-1] * 100:.2f}%")

# Count perfect epochs
perfect_val_epochs = df[df['val_acc'] >= 0.9999].shape[0]
print(f"\nEpochs with 100% validation accuracy: {perfect_val_epochs}/{len(df)}")

print("\n" + "=" * 70)
print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nFiles saved to: {output_dir}")
print("  1. training_loss.png")
print("  2. training_accuracy.png")
print("  3. training_auc.png")
print("  4. learning_rate.png")
print("  5. training_curves_combined.png ⭐ (Use this for paper)")
print("\nReady for paper inclusion! 🎉")
print()
