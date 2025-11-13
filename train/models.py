"""
Neural network architecture components.
Implements dual-stream CNN + ConvLSTM + attention fusion for deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class SpatialCNN(nn.Module):
    """
    Spatial feature extractor using EfficientNet backbone.
    Processes each frame independently to extract spatial features. 
    """
    
    def __init__(self, model_name='efficientnet-b4', pretrained=True, freeze_early_layers=False):
        """
        Args:
            model_name: EfficientNet variant ('efficientnet-b0' to 'efficientnet-b7')
            pretrained: Load ImageNet pretrained weights
            freeze_early_layers: Freeze early conv layers for transfer learning
        """
        super().__init__()
        
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        # Remove classification head (keep feature extractor)
        self.backbone._fc = nn.Identity()
        
        # Get output feature dimension
        self.feature_dim = self.backbone._conv_head.out_channels
        
        # Optionally freeze early layers
        if freeze_early_layers:
            for name, param in self.backbone.named_parameters():
                if '_blocks.0' in name or '_blocks.1' in name or '_conv_stem' in name:
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B*T, 3, H, W) - batched frames
        
        Returns:
            features: (B*T, feature_dim, h, w) spatial features
        """
        # Extract features (before global pooling)
        x = self.backbone.extract_features(x)  # (B*T, C, h, w)
        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    Replaces matrix multiplications with convolutions to preserve spatial structure.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        """
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden state channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: Input (B, input_dim, H, W)
            h_prev: Previous hidden state (B, hidden_dim, H, W)
            c_prev: Previous cell state (B, hidden_dim, H, W)
        
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        gates = self.conv(combined)
        
        # Split into 4 gates
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell and hidden states
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_size):
        """Initialize hidden and cell states with zeros."""
        h, w = spatial_size
        device = self.conv.weight.device
        
        h_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        c_state = torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        
        return h_state, c_state


class ConvLSTMModule(nn.Module):
    """
    Stacked ConvLSTM for temporal modeling.
    Processes sequence of spatial feature maps and outputs temporal representation.
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128], kernel_size=3, output_dim=512):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions for each layer
            kernel_size: Convolution kernel size
            output_dim: Final output dimension after pooling
        """
        super().__init__()
        
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Stack of ConvLSTM cells
        self.cells = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cells.append(ConvLSTMCell(in_dim, hidden_dims[i], kernel_size))
        
        # Final projection to output_dim
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (B, T, C, H, W)
        
        Returns:
            output: Temporal representation (B, output_dim)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Initialize hidden states for all layers
        h_states = []
        c_states = []
        for i in range(self.num_layers):
            h, c = self.cells[i].init_hidden(batch_size, (H, W))
            h_states.append(h)
            c_states.append(c)
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t]  # (B, C, H, W)
            
            # Pass through stacked ConvLSTM layers
            for i in range(self.num_layers):
                h_states[i], c_states[i] = self.cells[i](x_t, h_states[i], c_states[i])
                x_t = h_states[i]  # Use output as input to next layer
        
        # Use final hidden state of last layer
        final_hidden = h_states[-1]  # (B, hidden_dims[-1], H, W)
        
        # Pool and project to output_dim
        output = self.projection(final_hidden)  # (B, output_dim)
        
        return output


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of HQ and LQ stream features.
    Learns to weight each stream based on content.
    """
    
    def __init__(self, feature_dim=512):
        """
        Args:
            feature_dim: Dimension of input features from each stream
        """
        super().__init__()
        
        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, hq_features, lq_features):
        """
        Args:
            hq_features: High-quality stream features (B, feature_dim)
            lq_features: Low-quality stream features (B, feature_dim)
        
        Returns:
            fused: Attention-weighted fusion (B, feature_dim)
        """
        # Compute attention scores
        score_hq = self.attention(hq_features)  # (B, 1)
        score_lq = self.attention(lq_features)  # (B, 1)
        
        # Softmax to get attention weights
        scores = torch.cat([score_hq, score_lq], dim=1)  # (B, 2)
        weights = F.softmax(scores, dim=1)  # (B, 2)
        
        alpha_hq = weights[:, 0:1]  # (B, 1)
        alpha_lq = weights[:, 1:2]  # (B, 1)
        
        # Weighted combination
        fused = alpha_hq * hq_features + alpha_lq * lq_features
        
        return fused


class DeepfakeDetector(nn.Module):
    """
    Complete deepfake detection model.
    Dual-stream (HQ/LQ) CNN + ConvLSTM + attention fusion + classifier.
    """
    
    def __init__(
        self,
        hq_backbone='efficientnet-b4',
        lq_backbone='efficientnet-b0',
        convlstm_hidden_dims=[256, 128],
        temporal_dim=512,
        dropout_rates=[0.5, 0.3],
        pretrained=True
    ):
        """
        Args:
            hq_backbone: EfficientNet model for HQ stream
            lq_backbone: EfficientNet model for LQ stream
            convlstm_hidden_dims: Hidden dimensions for ConvLSTM layers
            temporal_dim: Temporal feature dimension
            dropout_rates: Dropout rates for classifier layers
            pretrained: Use pretrained backbones
        """
        super().__init__()
        
        # HQ stream (224x224)
        self.hq_cnn = SpatialCNN(hq_backbone, pretrained=pretrained)
        self.hq_convlstm = ConvLSTMModule(
            input_dim=self.hq_cnn.feature_dim,
            hidden_dims=convlstm_hidden_dims,
            output_dim=temporal_dim
        )
        
        # LQ stream (112x112)
        self.lq_cnn = SpatialCNN(lq_backbone, pretrained=pretrained)
        self.lq_convlstm = ConvLSTMModule(
            input_dim=self.lq_cnn.feature_dim,
            hidden_dims=convlstm_hidden_dims,
            output_dim=temporal_dim
        )
        
        # Fusion
        self.fusion = AttentionFusion(feature_dim=temporal_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, hq_seq, lq_seq):
        """
        Args:
            hq_seq: HQ sequence (B, T, 3, 224, 224)
            lq_seq: LQ sequence (B, T, 3, 112, 112)
        
        Returns:
            logits: Prediction logits (B, 1)
        """
        batch_size, T, C, H_hq, W_hq = hq_seq.size()
        _, _, _, H_lq, W_lq = lq_seq.size()
        
        # Reshape for CNN: (B, T, C, H, W) -> (B*T, C, H, W)
        hq_frames = hq_seq.view(batch_size * T, C, H_hq, W_hq)
        lq_frames = lq_seq.view(batch_size * T, C, H_lq, W_lq)
        
        # Extract spatial features
        hq_spatial = self.hq_cnn(hq_frames)  # (B*T, C_hq, h, w)
        lq_spatial = self.lq_cnn(lq_frames)  # (B*T, C_lq, h, w)
        
        # Reshape back: (B*T, C, h, w) -> (B, T, C, h, w)
        _, C_hq, h_hq, w_hq = hq_spatial.size()
        _, C_lq, h_lq, w_lq = lq_spatial.size()
        
        hq_spatial = hq_spatial.view(batch_size, T, C_hq, h_hq, w_hq)
        lq_spatial = lq_spatial.view(batch_size, T, C_lq, h_lq, w_lq)
        
        # Temporal modeling with ConvLSTM
        hq_temporal = self.hq_convlstm(hq_spatial)  # (B, temporal_dim)
        lq_temporal = self.lq_convlstm(lq_spatial)  # (B, temporal_dim)
        
        # Fusion
        fused = self.fusion(hq_temporal, lq_temporal)  # (B, temporal_dim)
        
        # Classification
        logits = self.classifier(fused)  # (B, 1)
        
        return logits


def create_model(config):
    """
    Factory function to create model from config dict.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        DeepfakeDetector: Model instance
    """
    model = DeepfakeDetector(
        hq_backbone=config.get('hq_backbone', 'efficientnet-b4'),
        lq_backbone=config.get('lq_backbone', 'efficientnet-b0'),
        convlstm_hidden_dims=config.get('convlstm_filters', [256, 128]),
        temporal_dim=512,
        dropout_rates=config.get('dropout', [0.5, 0.3]),
        pretrained=True
    )
    
    return model
