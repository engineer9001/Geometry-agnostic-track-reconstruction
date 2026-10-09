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
    """Positional encoding for channel positions.

    A single shared table is used for BOTH straw hits and calorimeter crystals.
    To avoid PE collisions between "straw channel 100" and "crystal 100" the
    dataset offsets crystal indices by CRYSTAL_PE_OFFSET before they are
    passed in (see MaskedTransformerEncoder for the constant).  The detector
    identity is *also* encoded by a learned detector-type embedding, giving
    the model two independent cues to distinguish tracker from calo tokens.
    """

    def __init__(self, d_model: int, max_channels: int = 50000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_channels = max_channels

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
            channel_indices: (batch, seq_len) containing detector-element IDs
                             (straws in [0, CRYSTAL_PE_OFFSET); crystals with
                             the offset already applied).
        """
        if channel_indices is not None:
            pe_expanded = self.pe.squeeze(0)
            # Clamp defensively so an out-of-range index cannot index-error
            # under AMP; the detector-type embedding still disambiguates.
            safe_idx = channel_indices.clamp(0, self.max_channels - 1)
            pe_gathered = pe_expanded[safe_idx]
            return self.dropout(x + pe_gathered)
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ---------------------------------------------------------------------------
# Tracker per-hit standardization.
#
# New feature layout (4 features):
#   t0, t1  in [ns], absolute-from-earliest-hit-in-track
#   tot0, tot1  in [ns/counts], time-over-threshold at each wire end.
#
# t0/t1 statistics come from the earlier probe of MLData/FlatTrainh5/;
# tot0/tot1 are placeholders based on the mu2e reco nominal scale
# (per-end TOT is roughly uniform in [0, ~50] ns with mean ~20 ns and
# std ~10 ns).  Re-run probe_ml_inputs.py against the regenerated flat
# files to lock these in.
# ---------------------------------------------------------------------------
_DEFAULT_INPUT_MEAN = (22.7646, 22.7737, 20.0, 20.0)
_DEFAULT_INPUT_STD  = (14.0585, 14.0667, 10.0, 10.0)
_INPUT_CLAMP        = 8.0
_NAN_SENTINEL       = -5.0   # standardized-unit sentinel for NaN t0/t1


# ---------------------------------------------------------------------------
# Per-crystal calo standardization.
#
# The network's ONLY view of the calorimeter is now the per-crystal hit
# block  [edep, time]  (crystal_id is used only for positional encoding).
# This deliberately removes the earlier target-leak problem where the head
# was being handed reconstructed track momentum at the calo face via the
# 22-vector calo_scalars.
#
# Rough physical scales -- refine with a probe once new flat files exist:
#   crystal edep  ~ 0-60 MeV per hit; mean ~5 MeV, std ~10 MeV
#   crystal time  ~ 0-150 ns (same clock as straw t0/t1, anchored to the
#                  earliest hit in the track); mean ~50 ns, std ~30 ns
# Order matches the crystal feature layout after crystal_id is stripped in
# the dataset:  [edep, time].
# ---------------------------------------------------------------------------
_DEFAULT_CRYSTAL_MEAN = (5.0, 50.0)
_DEFAULT_CRYSTAL_STD  = (10.0, 30.0)
_CRYSTAL_CLAMP        = 8.0
CRYSTAL_INPUT_DIM     = 2      # edep, time

# Offset applied to raw crystal_id before shared-PE lookup.  Straws occupy
# [0, 41472) in the standard mu2e index; 42000 gives a safety margin.
CRYSTAL_PE_OFFSET     = 42000


class MaskedTransformerEncoder(nn.Module):
    """
    Straw-hit transformer encoder with optional per-crystal calorimeter tokens
    fused into the same attention stack.

    When crystal inputs are provided:
      * crystals are projected through their own Linear (different feature
        dim and physical scale than straws) and normalized with a shared
        LayerNorm(d_model);
      * a learned detector-type embedding (0=straw, 1=crystal) is added to
        every token;
      * the shared positional encoding is used, with crystal indices offset
        by CRYSTAL_PE_OFFSET so straw-channel and crystal-id spaces do not
        collide;
      * straw and crystal tokens are concatenated along the sequence axis
        and passed through the standard TransformerEncoder with a combined
        key-padding mask.

    When no crystal inputs are provided the module behaves exactly like the
    original tracker-only encoder.
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_channels: int = 50000,
        input_mean: Optional[Tuple[float, ...]] = None,
        input_std: Optional[Tuple[float, ...]] = None,
        use_crystals: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_crystals = use_crystals

        # ---- straw input path ----
        if input_mean is None:
            input_mean = _DEFAULT_INPUT_MEAN[:input_dim]
        if input_std is None:
            input_std = _DEFAULT_INPUT_STD[:input_dim]
        if len(input_mean) != input_dim or len(input_std) != input_dim:
            raise ValueError(
                f"input_mean/input_std length ({len(input_mean)}, {len(input_std)}) "
                f"must match input_dim ({input_dim})"
            )
        self.register_buffer(
            "_input_mean", torch.tensor(input_mean, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer(
            "_input_std", torch.tensor(input_std, dtype=torch.float32).view(1, 1, -1)
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # ---- crystal input path (built even when use_crystals=False so a
        #      checkpoint saved without calo can be re-loaded with calo
        #      enabled by simply flipping the flag; the unused parameters
        #      have negligible parameter count) ----
        self.register_buffer(
            "_crystal_mean",
            torch.tensor(_DEFAULT_CRYSTAL_MEAN, dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "_crystal_std",
            torch.tensor(_DEFAULT_CRYSTAL_STD, dtype=torch.float32).view(1, 1, -1),
        )
        self.crystal_projection = nn.Linear(CRYSTAL_INPUT_DIM, d_model)
        self.crystal_norm = nn.LayerNorm(d_model)

        # Detector-type embedding.  Even with the PE offset the model has to
        # know "this is a calo token" to route it through the right internal
        # circuit; a 2-row learned embedding is the cheapest way to encode
        # that categorical fact.
        self.detector_type_emb = nn.Embedding(2, d_model)

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

    # -- helpers ---------------------------------------------------------
    def _standardize_straws(self, x: torch.Tensor) -> torch.Tensor:
        bad_mask: Optional[torch.Tensor] = None
        if not torch.isfinite(x).all():
            bad_mask = ~torch.isfinite(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self._input_mean) / self._input_std
        x = torch.clamp(x, -_INPUT_CLAMP, _INPUT_CLAMP)
        if bad_mask is not None:
            x = torch.where(bad_mask, torch.full_like(x, _NAN_SENTINEL), x)
        return x

    def _standardize_crystals(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self._crystal_mean) / self._crystal_std
        x = torch.clamp(x, -_CRYSTAL_CLAMP, _CRYSTAL_CLAMP)
        return x

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
        crystal_x: Optional[torch.Tensor] = None,
        crystal_mask: Optional[torch.Tensor] = None,
        crystal_ch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:               (B, S, input_dim)  straw features [t0, t1, edep]
            src_key_padding_mask: (B, S)  True = padding
            channel_indices: (B, S)       straw channel IDs (unshifted)
            crystal_x:       (B, C, 2)    per-crystal features [edep, time] or None
            crystal_mask:    (B, C)       True = padding, or None
            crystal_ch:      (B, C)       raw crystal_id (unshifted); the
                                          offset is applied inside forward.

        Returns:
            encoded:        (B, S+C, d_model)
            combined_mask:  (B, S+C)  padding mask for the returned sequence
                            (or None if none was supplied and no crystals).
        """
        # -- straw branch --
        x_std   = self._standardize_straws(x)
        s_emb   = self.input_norm(self.input_projection(x_std))
        s_emb   = self.pos_encoding(s_emb, channel_indices=channel_indices)
        # Add detector-type embedding (index 0 = straw).  Using .weight[0]
        # avoids allocating an intermediate index tensor every step.
        s_emb   = s_emb + self.detector_type_emb.weight[0].view(1, 1, -1)

        if crystal_x is None or not self.use_crystals:
            encoded = self.transformer_encoder(
                s_emb, src_key_padding_mask=src_key_padding_mask
            )
            return encoded, src_key_padding_mask

        # -- crystal branch --
        c_std = self._standardize_crystals(crystal_x)
        c_emb = self.crystal_norm(self.crystal_projection(c_std))
        # Shift crystal IDs into the reserved PE region so straws and
        # crystals do not share PE rows.
        if crystal_ch is None:
            raise ValueError("crystal_ch must be provided when use_crystals=True")
        c_ch_shifted = crystal_ch + CRYSTAL_PE_OFFSET
        c_emb = self.pos_encoding(c_emb, channel_indices=c_ch_shifted)
        c_emb = c_emb + self.detector_type_emb.weight[1].view(1, 1, -1)

        # -- fuse and encode --
        combined = torch.cat([s_emb, c_emb], dim=1)
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )
        if crystal_mask is None:
            crystal_mask = torch.zeros(
                crystal_x.shape[0], crystal_x.shape[1],
                dtype=torch.bool, device=crystal_x.device,
            )
        combined_mask = torch.cat([src_key_padding_mask, crystal_mask], dim=1)

        # Guard against a track whose ENTIRE combined sequence is padding
        # (shouldn't happen — every track has ≥5 straw hits after the flat
        # cut — but a fully-True row would produce NaN inside softmax).
        if combined_mask.all(dim=1).any():
            # Unmask the first token of any such row.
            all_pad = combined_mask.all(dim=1)
            combined_mask = combined_mask.clone()
            combined_mask[all_pad, 0] = False

        encoded = self.transformer_encoder(
            combined, src_key_padding_mask=combined_mask
        )
        return encoded, combined_mask


# ---------------------------------------------------------------------------
# Pooling helper shared by the two momentum heads.  Kept as a free function
# so the heads remain small and readable.
# ---------------------------------------------------------------------------
def _pool(
    encoded: torch.Tensor,
    mask: Optional[torch.Tensor],
    pooling_type: str,
    attention_pooling: Optional[nn.MultiheadAttention] = None,
    query_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pooling_type == "mean":
        if mask is not None:
            real_mask_expanded = (~mask).unsqueeze(-1).float()
            return (encoded * real_mask_expanded).sum(dim=1) / (
                real_mask_expanded.sum(dim=1).clamp(min=1.0)
            )
        return encoded.mean(dim=1)
    if pooling_type == "max":
        if mask is not None:
            return encoded.masked_fill(mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        return encoded.max(dim=1).values
    if pooling_type == "attention":
        batch_size = encoded.size(0)
        query = query_token.expand(batch_size, -1, -1)
        attn_out, _ = attention_pooling(query, encoded, encoded, key_padding_mask=mask)
        return attn_out.squeeze(1)
    raise ValueError(f"Unknown pooling_type: {pooling_type}")


class AbsMomentumPredictionHead(nn.Module):
    """
    Pools the transformer output and predicts (pT, |pz|).  All calo info
    (if any) is now supplied via extra tokens in the transformer, so this
    head no longer takes a `calo_scalars` argument.
    """

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
        else:
            self.attention_pooling = None
            self.query_token = None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 2),  # pT, |pz|
        )

    def forward(self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = _pool(encoded, mask, self.pooling_type,
                       self.attention_pooling, self.query_token)
        return self.mlp(pooled)


class BinaryClassificationHead(nn.Module):
    """Masked-pool track tokens and emit one raw Ce-vs-DIO logit per track."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attention_pooling = nn.MultiheadAttention(
                d_model, num_heads=1, batch_first=True, dropout=dropout
            )
            self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
        else:
            self.attention_pooling = None
            self.query_token = None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )

    def forward(self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = _pool(
            encoded, mask, self.pooling_type,
            self.attention_pooling, self.query_token,
        )
        return self.mlp(pooled).squeeze(-1)


class MomentumPredictionHead(nn.Module):
    """
    Pools the transformer output and predicts a 3D momentum vector.
    Calo information (if enabled) enters via extra tokens in the transformer,
    so this head no longer takes a `calo_scalars` argument.
    """

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
        else:
            self.attention_pooling = None
            self.query_token = None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 3),  # Px, Py, Pz
        )

    def forward(self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = _pool(encoded, mask, self.pooling_type,
                       self.attention_pooling, self.query_token)
        return self.mlp(pooled)


# ---------------------------------------------------------------------------
# Tracker geometry constants used by the CvN stem.
#
# Channel packing (see track_aggregation.py / preprocess_to_flat.py):
#   ch = ((plane * N_PANEL + panel) * N_LAYER + layer) * N_STRAW + straw
# so
#   straw = ch % N_STRAW
#   layer = (ch // N_STRAW) % N_LAYER
#   panel = (ch // (N_STRAW * N_LAYER)) % N_PANEL
#   plane =  ch // (N_STRAW * N_LAYER * N_PANEL)
#
# Panel is cyclic (0..5 → wraps mod 6); the geometry embedding for Δpanel
# therefore uses (sin, cos) of the wrapped angle rather than a raw integer
# delta.  Plane and straw are linear; layer is binary.
# ---------------------------------------------------------------------------
N_PLANE = 36
N_PANEL = 6
N_LAYER = 2
N_STRAW = 96
N_STRAW_CHANNELS = N_PLANE * N_PANEL * N_LAYER * N_STRAW  # 41472


def _decode_straw_channel(ch: torch.Tensor):
    """
    Decode a packed straw channel index into (plane, panel, layer, straw).

    Args:
        ch: (B, S) int64 tensor of packed straw channel IDs.

    Returns:
        plane, panel, layer, straw: four (B, S) int64 tensors.
    """
    straw = ch % N_STRAW
    layer = (ch // N_STRAW) % N_LAYER
    panel = (ch // (N_STRAW * N_LAYER)) % N_PANEL
    plane =  ch // (N_STRAW * N_LAYER * N_PANEL)
    return plane, panel, layer, straw


class ConvNeighborStem(nn.Module):
    """
    Geometry-aware convolutional token embedder.

    For each real straw hit ("anchor"), this module finds the K nearest real
    hits inside the same track under a learned geometry-aware distance in
    (plane, panel_cyclic, layer, straw) space and produces a mixed embedding
    of shape (B, S, d_model).

    Rationale.  A dense Conv4D over the tracker voxel grid
    (36 × 6 × 2 × 96 = 41 472 cells) would waste ~99.9 % of its FLOPs
    (typical occupancy is a few tens of hits per track) and cannot cleanly
    give the four axes their proper treatments (linear on plane/straw,
    cyclic on panel, degenerate on layer=2).  A sparse local mixer over the
    K nearest actual hits gives the same weight sharing without any of that
    waste.

    Output is a per-hit embedding at d_model, ready to be added to the
    positional encoding and detector-type embedding by the outer encoder.
    Padded hits produce a zero embedding.
    """

    # Geometric-delta feature vector per neighbor pair:
    #   [Δplane / N_PLANE,
    #    sin(2π Δpanel / N_PANEL), cos(2π Δpanel / N_PANEL),
    #    Δlayer,                                  # in {-1, 0, 1}
    #    Δstraw / N_STRAW,
    #    same-plane, same-panel, same-layer      # binary indicators
    #   ]
    GEOM_DIM = 8

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        k_neighbors: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.k = k_neighbors

        # Anchor path: identical role to the old input_projection — a per-hit
        # linear map from raw features to d_model.  Retained under the same
        # name so residual reasoning about the model is easier.
        self.input_projection = nn.Linear(input_dim, d_model)

        # Local mixer: sees the concatenation of (anchor features, neighbor
        # features, geometric delta) and produces a d_model contribution.
        mix_in = 2 * input_dim + self.GEOM_DIM
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(mix_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Learned positive weights on each axis of the distance used for
        # neighbor selection.  Initialized so all four axes contribute
        # roughly equally after the softplus.
        self._raw_axis_weights = nn.Parameter(torch.zeros(4))

        # Output normalization on the summed (anchor + mixed neighbors)
        # embedding.  Plays the role of the old input_norm.
        self.output_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    def _axis_weights(self) -> torch.Tensor:
        # softplus + small constant to guarantee strictly positive weights.
        return F.softplus(self._raw_axis_weights) + 1e-3

    # ------------------------------------------------------------------
    def _pairwise_distance(
        self,
        plane: torch.Tensor,   # (B, S) long
        panel: torch.Tensor,   # (B, S) long
        layer: torch.Tensor,   # (B, S) long
        straw: torch.Tensor,   # (B, S) long
        pad_mask: torch.Tensor,  # (B, S) bool, True=padding
    ) -> torch.Tensor:
        """
        Return the (B, S, S) pairwise "geometry distance" matrix used for
        neighbor selection.  Padded columns are set to +inf so they are
        never chosen; the diagonal is also +inf so a hit isn't its own
        neighbor.

        This block is *forced* to fp32 regardless of the outer autocast
        context.  In fp16, mixing `+inf` masked fills with learned axis
        weights near zero produced sporadic NaNs (a `0 * +inf` can escape
        into the topk selection under autocast).  The distance tensor is
        only used for a non-differentiable top-K selection anyway, so
        keeping it in fp32 has no gradient/accuracy cost.
        """
        # Force fp32 for numerical stability; distances feed a non-diff top-K.
        with torch.autocast(device_type=plane.device.type, enabled=False):
            # Cast to float32 once; keep on the same device.
            p  = plane.to(torch.float32)
            pa = panel.to(torch.float32)
            l  = layer.to(torch.float32)
            s  = straw.to(torch.float32)

            # (B, S, 1) vs (B, 1, S)
            dp = (p.unsqueeze(2) - p.unsqueeze(1)).abs() / float(N_PLANE)
            dl = (l.unsqueeze(2) - l.unsqueeze(1)).abs() / float(N_LAYER)
            ds = (s.unsqueeze(2) - s.unsqueeze(1)).abs() / float(N_STRAW)

            # Panel is cyclic: use 1 - cos(2π Δ / N_PANEL) as a smooth,
            # rotation-invariant distance in [0, 2].
            d_panel_ang = (pa.unsqueeze(2) - pa.unsqueeze(1)) * (2.0 * math.pi / float(N_PANEL))
            dpa = 0.5 * (1.0 - torch.cos(d_panel_ang))  # in [0, 1]

            w = self._axis_weights().to(torch.float32)  # (4,) positive
            dist = w[0] * dp + w[1] * dpa + w[2] * dl + w[3] * ds

            # Mask out self and padded neighbors.
            B, S = plane.shape
            eye = torch.eye(S, dtype=torch.bool, device=plane.device).unsqueeze(0)
            dist = dist.masked_fill(eye, float("inf"))
            # (B, 1, S) — a column j is padding => that neighbor is unusable.
            dist = dist.masked_fill(pad_mask.unsqueeze(1), float("inf"))
        return dist

    # ------------------------------------------------------------------
    def _geometry_features(
        self,
        anchor_plane: torch.Tensor,  # (B, S, 1) long
        anchor_panel: torch.Tensor,
        anchor_layer: torch.Tensor,
        anchor_straw: torch.Tensor,
        nbr_plane: torch.Tensor,     # (B, S, K) long
        nbr_panel: torch.Tensor,
        nbr_layer: torch.Tensor,
        nbr_straw: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-neighbor geometric delta feature vector."""
        dp = (nbr_plane - anchor_plane).float() / float(N_PLANE)
        dl = (nbr_layer - anchor_layer).float()  # in {-1, 0, 1}
        ds = (nbr_straw - anchor_straw).float() / float(N_STRAW)

        # Panel: cyclic angle → (sin, cos) of the shortest signed delta.
        dpa_ang = (nbr_panel - anchor_panel).float() * (2.0 * math.pi / float(N_PANEL))
        sin_pa = torch.sin(dpa_ang)
        cos_pa = torch.cos(dpa_ang)

        same_plane = (nbr_plane == anchor_plane).float()
        same_panel = (nbr_panel == anchor_panel).float()
        same_layer = (nbr_layer == anchor_layer).float()

        return torch.stack(
            [dp, sin_pa, cos_pa, dl, ds, same_plane, same_panel, same_layer],
            dim=-1,
        )  # (B, S, K, GEOM_DIM)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_std: torch.Tensor,           # (B, S, input_dim), already standardized
        channel_indices: torch.Tensor, # (B, S) int64
        pad_mask: torch.Tensor,        # (B, S) bool, True=padding
    ) -> torch.Tensor:
        B, S, F_in = x_std.shape
        device = x_std.device

        # Handle tracks with fewer than 2 real hits (or trivially short
        # sequences) by falling back to a pure linear projection — there
        # are no neighbors to mix in.
        if S < 2:
            return self.output_norm(self.input_projection(x_std))

        # Effective K cannot exceed S - 1 (each hit has at most S-1 others
        # in the batch element).  Use a single global K for the whole batch
        # so the mixer MLP sees a uniform tensor shape; padding within an
        # individual track is handled by the +inf distances below.
        k = min(self.k, S - 1)

        # --- decode geometry ---
        plane, panel, layer, straw = _decode_straw_channel(channel_indices)

        # Zero-out geometry for padded slots so they can't accidentally act
        # as "close to the origin" attractors.  (They also have +inf column
        # distance so this is belt-and-braces.)
        # (No-op numerically, but keeps intermediates well-defined.)

        # --- distances and top-K neighbor selection ---
        # For fully-padded rows (an anchor that is itself padding) all
        # distances are still valid on the row axis; we simply won't use
        # the output at those positions (the caller masks them out).
        dist = self._pairwise_distance(plane, panel, layer, straw, pad_mask)  # (B, S, S)

        # `largest=False` picks the K smallest distances per anchor row.
        # Neighbors that were masked to +inf will still be returned when
        # the track has fewer than K+1 real hits; we detect and discard
        # those pairs via a validity mask built from dist.
        nbr_dist, nbr_idx = torch.topk(dist, k=k, dim=-1, largest=False)  # (B, S, K)
        valid_nbr = torch.isfinite(nbr_dist)  # (B, S, K)

        # --- gather neighbor features & geometry ---
        # Expand nbr_idx to gather along the feature and coord axes.
        idx_feat  = nbr_idx.unsqueeze(-1).expand(-1, -1, -1, F_in)          # (B, S, K, F)
        # x_std is (B, S, F); we need to gather from dim=1 for each anchor.
        # torch.gather requires the indexed dim to be the same shape as the
        # source in every OTHER dim, so we expand x_std to (B, S, S, F).
        # To avoid the S² blow-up in feature dim, use advanced indexing.
        batch_ar = torch.arange(B, device=device).view(B, 1, 1).expand(B, S, k)  # (B, S, K)
        nbr_feats = x_std[batch_ar, nbr_idx]                                     # (B, S, K, F)

        nbr_plane_g = plane[batch_ar, nbr_idx]  # (B, S, K)
        nbr_panel_g = panel[batch_ar, nbr_idx]
        nbr_layer_g = layer[batch_ar, nbr_idx]
        nbr_straw_g = straw[batch_ar, nbr_idx]

        geom = self._geometry_features(
            plane.unsqueeze(-1), panel.unsqueeze(-1),
            layer.unsqueeze(-1), straw.unsqueeze(-1),
            nbr_plane_g, nbr_panel_g, nbr_layer_g, nbr_straw_g,
        )  # (B, S, K, GEOM_DIM)

        # Anchor features broadcast to (B, S, K, F).
        anchor_feats = x_std.unsqueeze(2).expand(-1, -1, k, -1)

        pair_in = torch.cat([anchor_feats, nbr_feats, geom], dim=-1)  # (B, S, K, 2F+GEOM)
        pair_out = self.neighbor_mlp(pair_in)                          # (B, S, K, d_model)

        # Mean-pool over valid neighbors only.
        valid_f = valid_nbr.unsqueeze(-1).float()
        pair_sum = (pair_out * valid_f).sum(dim=2)                    # (B, S, d_model)
        n_valid  = valid_f.sum(dim=2).clamp(min=1.0)                  # (B, S, 1)
        nbr_ctx  = pair_sum / n_valid

        # Anchor's own linear projection + local neighborhood mixing.
        anchor_emb = self.input_projection(x_std)                     # (B, S, d_model)
        emb = self.output_norm(anchor_emb + nbr_ctx)

        # Zero-out padded anchors so they don't inject arbitrary values
        # into any subsequent LayerNorm statistics if pooling ever
        # unmasks them by mistake.
        emb = emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return emb


class CvnMaskedTransformerEncoder(MaskedTransformerEncoder):
    """
    Masked transformer encoder whose *straw* input path uses the
    geometry-aware ConvNeighborStem in place of the plain
    Linear(input_dim → d_model) + LayerNorm.

    The crystal path, positional encoding, detector-type embedding, and
    downstream transformer stack are inherited from MaskedTransformerEncoder
    unchanged.
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_channels: int = 50000,
        input_mean: Optional[Tuple[float, ...]] = None,
        input_std: Optional[Tuple[float, ...]] = None,
        use_crystals: bool = False,
        k_neighbors: int = 6,
    ):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_channels=max_channels,
            input_mean=input_mean,
            input_std=input_std,
            use_crystals=use_crystals,
        )
        # Replace the plain per-hit projection with the neighbor-conv stem.
        # We keep the inherited input_projection/input_norm modules on the
        # module tree (harmless dead parameters) so that older checkpoints
        # with matching state_dict keys can still be partially loaded; the
        # forward path below does not call them.
        self.conv_stem = ConvNeighborStem(
            input_dim=input_dim,
            d_model=d_model,
            k_neighbors=k_neighbors,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
        crystal_x: Optional[torch.Tensor] = None,
        crystal_mask: Optional[torch.Tensor] = None,
        crystal_ch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # -- straw branch (CvN stem) --
        if channel_indices is None:
            raise ValueError(
                "CvnMaskedTransformerEncoder requires channel_indices; "
                "the geometry-aware stem cannot operate without them."
            )
        if src_key_padding_mask is None:
            pad_mask = torch.zeros(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )
        else:
            pad_mask = src_key_padding_mask

        x_std = self._standardize_straws(x)
        s_emb = self.conv_stem(x_std, channel_indices, pad_mask)
        s_emb = self.pos_encoding(s_emb, channel_indices=channel_indices)
        s_emb = s_emb + self.detector_type_emb.weight[0].view(1, 1, -1)

        if crystal_x is None or not self.use_crystals:
            # Same all-pad-row guard as the base class applies to the
            # crystal-fused path: a fully-padded row would produce a NaN
            # inside softmax.  The track-quality cut is ≥5 hits so this
            # shouldn't fire in practice, but the guard is cheap.
            if src_key_padding_mask is not None and src_key_padding_mask.all(dim=1).any():
                all_pad = src_key_padding_mask.all(dim=1)
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_pad, 0] = False
            encoded = self.transformer_encoder(
                s_emb, src_key_padding_mask=src_key_padding_mask
            )
            return encoded, src_key_padding_mask

        # -- crystal branch (identical to the base class) --
        c_std = self._standardize_crystals(crystal_x)
        c_emb = self.crystal_norm(self.crystal_projection(c_std))
        if crystal_ch is None:
            raise ValueError("crystal_ch must be provided when use_crystals=True")
        c_ch_shifted = crystal_ch + CRYSTAL_PE_OFFSET
        c_emb = self.pos_encoding(c_emb, channel_indices=c_ch_shifted)
        c_emb = c_emb + self.detector_type_emb.weight[1].view(1, 1, -1)

        combined = torch.cat([s_emb, c_emb], dim=1)
        if crystal_mask is None:
            crystal_mask = torch.zeros(
                crystal_x.shape[0], crystal_x.shape[1],
                dtype=torch.bool, device=crystal_x.device,
            )
        combined_mask = torch.cat([pad_mask, crystal_mask], dim=1)

        if combined_mask.all(dim=1).any():
            all_pad = combined_mask.all(dim=1)
            combined_mask = combined_mask.clone()
            combined_mask[all_pad, 0] = False

        encoded = self.transformer_encoder(
            combined, src_key_padding_mask=combined_mask
        )
        return encoded, combined_mask


class TrackReconstructionModel(nn.Module):
    """
    Top-level model.

    When `use_calo=True` the encoder consumes per-crystal calorimeter tokens
    (edep, time) alongside straw hits.  When `use_calo=False` the model is
    identical to the tracker-only architecture.

    The legacy `calo_dim` kwarg is accepted for backwards compatibility but
    is ignored: the current design integrates calo through the transformer
    sequence, not through a post-pool concat.
    """

    def __init__(
        self,
        input_dim: int = 3,
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
        use_calo: bool = False,
        calo_dim: int = 0,  # legacy; ignored
    ):
        super().__init__()
        self.task = task
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_channels = max_channels
        self.use_calo = use_calo

        # The CvN variant swaps only the straw-token embedding stage for a
        # geometry-aware neighbor conv; everything downstream (PE, detector
        # type embedding, transformer stack, heads) is identical.
        if task in ("cvn_momentum", "cvn_classifier"):
            self.encoder = CvnMaskedTransformerEncoder(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                max_channels=max_channels,
                use_crystals=use_calo,
            )
        else:
            self.encoder = MaskedTransformerEncoder(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                max_channels=max_channels,
                use_crystals=use_calo,
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
        elif task in ("abs_momentum", "cvn_momentum"):
            # cvn_momentum shares the (pT, |pz|) output space with
            # abs_momentum so the geometry-aware stem is trained against a
            # target that already resolves the forward/backward and
            # φ-rotational degeneracies of the tracker.
            self.head = AbsMomentumPredictionHead(
                d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pooling_type=pooling_type,
            )
        elif task == "cvn_classifier":
            self.head = BinaryClassificationHead(
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
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
        crystal_x: Optional[torch.Tensor] = None,
        crystal_mask: Optional[torch.Tensor] = None,
        crystal_ch: Optional[torch.Tensor] = None,
        # Legacy kwarg, silently ignored so old training scripts still run:
        calo_scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded, combined_mask = self.encoder(
            x,
            src_key_padding_mask=mask,
            channel_indices=channel_indices,
            crystal_x=crystal_x,
            crystal_mask=crystal_mask,
            crystal_ch=crystal_ch,
        )

        if self.task in ("momentum", "abs_momentum", "cvn_momentum", "cvn_classifier"):
            return self.head(encoded, mask=combined_mask)
        return self.head(encoded)


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


# ---------------------------------------------------------------------------
# Momentum-target normalization.
#
# The previous scheme divided every component by a single scalar
# _MOMENTUM_SCALE = 105 MeV/c, which is roughly |p| for the CE endpoint.
# That keeps the target magnitude O(1) but leaves each component with a
# per-axis standard deviation of only ~0.58 in normalized units, so the
# MSE surface is quite flat.  The regressor also has no bias/scale
# separation between components.
#
# We now standardize each component independently:
#     target_norm = (target - mean) / std
# The optimizer therefore always sees a target with zero mean and unit
# variance per axis, which improves MSE conditioning and lets a plain L2
# criterion behave like an equally-weighted per-axis loss regardless of
# the physical unit of that axis.
#
# For the isotropic CE MC (|p| ≈ 104.97 MeV/c, uniform direction), each
# Cartesian component has mean 0 and std = |p|/√3 ≈ 60.62 MeV/c.
# For pT the mean is |p|·π/4 ≈ 82.4 MeV/c and the std is ≈ 32.0 MeV/c;
# for |pz| the mean is |p|/2 ≈ 52.5 MeV/c and the std is ≈ 30.3 MeV/c.
# Refine these numbers by running the analyze_distributions probe on the
# regenerated flat files.
# ---------------------------------------------------------------------------
_MOMENTUM_SCALE = 105.0                # kept for backwards compat / reporting

# (px, py, pz) — MeV/c
_MOMENTUM_MEAN = (0.0, 0.0, 0.0)
_MOMENTUM_STD  = (60.62, 60.62, 60.62)

# (pT, |pz|) — MeV/c
_ABS_MOMENTUM_MEAN = (82.4, 52.5)
_ABS_MOMENTUM_STD  = (32.0, 30.3)


def _standardize(target: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=target.dtype, device=target.device)
    s = torch.tensor(std,  dtype=target.dtype, device=target.device)
    return (target - m) / s


def momentum_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on per-component standardized (px, py, pz).

    Both `output` and `target` are assumed to be in physical units [MeV/c];
    the standardization is applied inside the loss so the model head still
    predicts in physical units and no external un-scaling is needed at
    evaluation time.
    """
    out_n    = _standardize(output, _MOMENTUM_MEAN, _MOMENTUM_STD)
    target_n = _standardize(target, _MOMENTUM_MEAN, _MOMENTUM_STD)
    return F.mse_loss(out_n, target_n)


def abs_momentum_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on per-component standardized (pT, |pz|).

    `target` arrives as the raw (px, py, pz); we form (pT, |pz|) inside
    the loss, then standardize.
    """
    px, py, pz = target[:, 0], target[:, 1], target[:, 2]
    pT = torch.sqrt(px**2 + py**2)
    abs_pz = torch.abs(pz)
    target_abs = torch.stack([pT, abs_pz], dim=1)

    out_n    = _standardize(output,     _ABS_MOMENTUM_MEAN, _ABS_MOMENTUM_STD)
    target_n = _standardize(target_abs, _ABS_MOMENTUM_MEAN, _ABS_MOMENTUM_STD)
    return F.mse_loss(out_n, target_n)


def masked_mse_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(output, target, reduction='none')
    loss[mask] = 0.0
    active_elements = (~mask).sum().clamp(min=1) * loss.shape[-1]
    return loss.sum() / active_elements


def masked_l1_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(output, target, reduction='none')
    loss[mask] = 0.0
    active_elements = (~mask).sum().clamp(min=1) * loss.shape[-1]
    return loss.sum() / active_elements