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

        # Pre-registered scale factors so feature scaling is a single
        # broadcast multiply with no extra tensor allocation per forward pass.
        if input_dim == 2:
            self.register_buffer("_feat_scale", torch.tensor([1.0 / 100.0, 1000.0]))
        else:
            self._feat_scale = None

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
        # Single broadcast multiply — no cat, no extra allocation.
        if self._feat_scale is not None:
            x_scaled = x * self._feat_scale
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
    """
    Pools the transformer output and predicts a 3D momentum vector.

    Tracker-only mode (calo_dim=0, default):
        MLP input = pooled tracker embedding (d_model,)
        Identical to the original behaviour — no calo path is touched.

    Tracker + calorimeter mode (calo_dim > 0):
        MLP input = [pooled tracker embedding | calo scalars]  (d_model + calo_dim,)
        The calo scalars are concatenated *after* pooling so the transformer
        architecture is unchanged.  NaN values (unmatched tracks) must be
        replaced with 0 by the caller; a calo_matched flag should be appended
        as the last element so the model can learn to ignore calo when absent.

    Switching between modes only requires changing calo_dim at construction
    time — no other code changes are needed.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling_type: str = "mean",
        calo_dim: int = 0,
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.d_model = d_model
        self.calo_dim = calo_dim

        if pooling_type == "attention":
            self.attention_pooling = nn.MultiheadAttention(
                d_model, num_heads=1, batch_first=True, dropout=dropout
            )
            self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        # MLP input width: tracker pooled embedding + optional calo scalars
        mlp_input_dim = d_model + calo_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 3),  # Px, Py, Pz
        )

    def forward(
        self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None,
        calo_scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoded:      (batch, seq_len, d_model)
            mask:         (batch, seq_len) True = padding
            calo_scalars: (batch, calo_dim) or None
                          Required when calo_dim > 0; ignored when calo_dim == 0.
        """
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
                # masked_fill avoids an extra clone() allocation
                pooled = encoded.masked_fill(mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            else:
                pooled = encoded.max(dim=1).values

        elif self.pooling_type == "attention":
            query = self.query_token.expand(batch_size, -1, -1)
            attn_out, _ = self.attention_pooling(
                query, encoded, encoded, key_padding_mask=mask
            )
            pooled = attn_out.squeeze(1)

        # Concatenate calo scalars when in tracker+calo mode.
        # When calo_dim == 0 this branch is never entered, so tracker-only
        # runs have zero overhead from this code path.
        if self.calo_dim > 0 and calo_scalars is not None:
            pooled = torch.cat([pooled, calo_scalars], dim=-1)

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
        calo_dim: int = 0,
    ):
        """
        Args:
            calo_dim: Number of calorimeter scalar features to concatenate after
                      pooling.  Set to 0 (default) for tracker-only training —
                      the model is then identical to the original architecture.
                      Set to len(CALO_SCALAR_COLS) + 1 (for the calo_matched flag)
                      when training with calorimeter data.
        """
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_channels = max_channels
        self.calo_dim = calo_dim

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
            # Reconstruction head does not use calo scalars (hit-level output)
            self.head = nn.Linear(d_model, input_dim)
        elif task == "momentum":
            self.head = MomentumPredictionHead(
                d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pooling_type=pooling_type,
                calo_dim=calo_dim,
            )
        elif task == "regression":
            self.head = nn.Sequential(
                nn.Linear(d_model + calo_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, output_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
        calo_scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (batch, seq_len, input_dim)
            mask:           (batch, seq_len) True = padding
            channel_indices:(batch, seq_len)
            calo_scalars:   (batch, calo_dim) or None
                            Pass None (or omit) for tracker-only inference.
                            Pass the NaN-replaced + calo_matched-appended tensor
                            for tracker+calo inference.
        """
        encoded = self.encoder(x, src_key_padding_mask=mask, channel_indices=channel_indices)

        if self.task == "momentum":
            output = self.head(encoded, mask=mask, calo_scalars=calo_scalars)
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
