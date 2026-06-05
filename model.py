import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class TrackModelConfig:
    """Configuration class for track models."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class PositionalEncoding(nn.Module):
    """Positional encoding for channel positions."""

    def __init__(self, d_model: int, max_channels: int = 50000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_channels, d_model)
        position = torch.arange(0, max_channels, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, channel_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            channel_indices: (batch, seq_len) containing original detector channel IDs
        Returns:
            x + positional encoding
        """
        if channel_indices is not None:
            # Extract the specific positional encodings for the active channels
            pe_expanded = self.pe.squeeze(0)  # (max_channels, d_model)
            pe_gathered = pe_expanded[channel_indices]  # (batch, seq_len, d_model)
            return self.dropout(x + pe_gathered)
        else:
            # Fallback to standard sequential positional encoding
            return self.dropout(x + self.pe[:, : x.size(1), :])


class MaskedTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_channels: int = 50000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_channels, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) tensor where:
               - x[..., 0] is t_diff (raw nanoseconds)
               - x[..., 1] is edep (raw MeV values)
            src_key_padding_mask: (batch, seq_len) boolean mask for sequence padding
            channel_indices: (batch, seq_len) unique detector channel identifier tokens
        """
        # Out-of-place feature scaling to bring raw physics scales into O(1) bounds
        if self.input_dim == 2:
            t_diff_scaled = x[..., 0:1] / 100.0
            edep_scaled = x[..., 1:2] * 1000.0
            x_scaled = torch.cat([t_diff_scaled, edep_scaled], dim=-1)
        else:
            x_scaled = x

        x_projected = self.input_projection(x_scaled)
        x_encoded = self.pos_encoding(x_projected, channel_indices=channel_indices)
        
        encoded = self.transformer_encoder(
            x_encoded, 
            src_key_padding_mask=src_key_padding_mask
        )
        return encoded


class MomentumPredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.d_model = d_model

        if pooling_type == "attention":
            self.attention_pooling = nn.MultiheadAttention(
                d_model, num_heads=1, batch_first=True, dropout=dropout
            )
            self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 3),  # Updated output dim from 1 to 3 for Px, Py, Pz
        )

    def forward(
        self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = encoded.size(0)

        if self.pooling_type == "mean":
            if mask is not None:
                real_mask = ~mask
                real_mask_expanded = real_mask.unsqueeze(-1).float()
                pooled = (encoded * real_mask_expanded).sum(dim=1) / (
                    real_mask_expanded.sum(dim=1).clamp(min=1.0)
                )
            else:
                pooled = encoded.mean(dim=1)

        elif self.pooling_type == "max":
            if mask is not None:
                real_mask = ~mask
                encoded_masked = encoded.clone()
                encoded_masked[mask] = float("-inf")
                pooled = encoded_masked.max(dim=1).values
            else:
                pooled = encoded.max(dim=1).values

        elif self.pooling_type == "attention":
            query = self.query_token.expand(batch_size, -1, -1)
            attn_out, _ = self.attention_pooling(
                query, encoded, encoded, key_padding_mask=mask
            )
            pooled = attn_out.squeeze(1)

        momentum_vector = self.mlp(pooled)
        return momentum_vector


class TrackReconstructionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_channels: int = 50000,
        task: str = "reconstruction",
        output_dim: Optional[int] = None,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_channels = max_channels

        self.encoder = MaskedTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_channels=max_channels,
        )

        if task == "reconstruction" or task == "denoising":
            self.head = nn.Linear(d_model, input_dim)
        elif task == "momentum":
            self.head = MomentumPredictionHead(
                d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pooling_type=pooling_type,
            )
        elif task == "regression":
            self.head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, output_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(x, src_key_padding_mask=mask, channel_indices=channel_indices)

        if self.task == "momentum":
            output = self.head(encoded, mask=mask)
        else:
            output = self.head(encoded)

        return output


class DenoisingTrackModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.noise_level = kwargs.pop("noise_level", 0.1)
        kwargs["task"] = "denoising"
        self.model = TrackReconstructionModel(**kwargs)

    def add_noise(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        noisy_x = x.clone()
        noise = torch.randn_like(x) * self.noise_level
        noisy_x[~mask] += noise[~mask]
        return torch.clamp(noisy_x, min=0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        channel_indices: Optional[torch.Tensor] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        x_noisy = self.add_noise(x, mask) if add_noise else x
        return self.model(x_noisy, mask=mask, channel_indices=channel_indices)


def momentum_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes MSE loss between predicted and true 3D momentum vectors (Px, Py, Pz).
    Expects both tensors to be shape (batch_size, 3).
    """
    return F.mse_loss(output, target)


def masked_mse_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes MSE loss only on the valid (unmasked) elements."""
    loss = F.mse_loss(output, target, reduction='none')
    loss[mask] = 0.0
    active_elements = (~mask).sum().clamp(min=1) * loss.shape[-1]
    return loss.sum() / active_elements


def masked_l1_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes L1 loss only on the valid (unmasked) elements."""
    loss = F.l1_loss(output, target, reduction='none')
    loss[mask] = 0.0
    active_elements = (~mask).sum().clamp(min=1) * loss.shape[-1]
    return loss.sum() / active_elements