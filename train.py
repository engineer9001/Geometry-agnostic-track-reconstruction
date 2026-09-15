import argparse
import json
import logging
import math
import os
import pickle
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import h5py
from tqdm import tqdm

from model import (
    TrackReconstructionModel,
    DenoisingTrackModel,
    TrackModelConfig,
    masked_mse_loss,
    masked_l1_loss,
    momentum_loss,
    abs_momentum_loss,
    CRYSTAL_PE_OFFSET,
)
from nedt import build_sparse_track_from_hdf5_group, infer_channel_index_from_hdf5


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def _resolve_data_paths(data_path: str) -> str:
    """Resolve data path: if directory, use as-is; if file, use file."""
    p = Path(data_path)
    if p.is_dir():
        h5_files = list(p.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {data_path}")
        return str(p)
    elif p.is_file():
        return str(p)
    else:
        raise FileNotFoundError(f"Path does not exist: {data_path}")


# ---------------------------------------------------------------------------
# Per-crystal padding helper.  Kept as a free function so the collate can
# stay flat and readable.
# ---------------------------------------------------------------------------
def _pad_crystals(
    crystal_feats_list: List[np.ndarray],
    crystal_ch_list:    List[np.ndarray],
    n_crys_list:        List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-length per-crystal blocks to a common length.

    Each item's arrays are ALREADY trimmed to the true number of crystals
    for that track.  Tracks with zero matched crystals arrive as empty
    arrays and are fully masked in the output.

    Returns:
        crystal_x    : (B, C_max, 2)  float32     [edep, time]
        crystal_ch   : (B, C_max)     long        raw crystal_ids (unshifted;
                                                  the model applies the PE
                                                  offset internally)
        crystal_mask : (B, C_max)     bool        True = padding
    """
    B = len(crystal_feats_list)
    C_max = max((int(n) for n in n_crys_list), default=0)
    # Guarantee at least length 1 so the transformer's sequence dim is
    # non-empty; the mask hides the padded row entirely.
    C_max = max(C_max, 1)

    crystal_x    = torch.zeros(B, C_max, 2, dtype=torch.float32)
    crystal_ch   = torch.zeros(B, C_max,    dtype=torch.long)
    crystal_mask = torch.ones (B, C_max,    dtype=torch.bool)   # True = pad

    for i, (feats, chs, n) in enumerate(
        zip(crystal_feats_list, crystal_ch_list, n_crys_list)
    ):
        n = int(n)
        if n == 0:
            continue
        crystal_x[i, :n]    = torch.from_numpy(feats)
        crystal_ch[i, :n]   = torch.from_numpy(chs)
        crystal_mask[i, :n] = False

    return crystal_x, crystal_ch, crystal_mask


def sparse_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Stitches variable-length sparse tracks into tightly packaged batch tensors.
    Pads only up to the longest sequence *inside this specific batch*.

    Supported tuple lengths returned by the datasets:
        2: (features, indices)                          — reconstruction
        3: (features, indices, momentum)                — momentum, tracker-only
        6: (features, indices, momentum,
            crystal_feats, crystal_ids, n_crys)         — momentum, tracker + per-crystal calo
    """
    n_items      = len(batch[0])
    has_momentum = n_items >= 3
    has_calo     = n_items == 6

    if has_calo:
        (features_list, indices_list, momentum_list,
         crystal_feats_list, crystal_ch_list, n_crys_list) = zip(*batch)
    elif has_momentum:
        features_list, indices_list, momentum_list = zip(*batch)
    else:
        features_list, indices_list = zip(*batch)

    batch_size = len(batch)
    max_hits = max(f.shape[0] for f in features_list)
    if max_hits == 0:
        max_hits = 1

    n_features = features_list[0].shape[1]

    x_padded       = torch.zeros((batch_size, max_hits, n_features), dtype=torch.float32)
    mask_padded    = torch.ones ((batch_size, max_hits),             dtype=torch.bool)
    indices_padded = torch.zeros((batch_size, max_hits),             dtype=torch.long)

    for i in range(batch_size):
        n_hits = features_list[i].shape[0]
        if n_hits > 0:
            x_padded[i, :n_hits]       = torch.from_numpy(features_list[i])
            mask_padded[i, :n_hits]    = False
            indices_padded[i, :n_hits] = torch.from_numpy(indices_list[i])

    if has_calo:
        crystal_x, crystal_ch, crystal_mask = _pad_crystals(
            list(crystal_feats_list), list(crystal_ch_list), list(n_crys_list)
        )
        return (x_padded, mask_padded, indices_padded,
                torch.stack(momentum_list),
                crystal_x, crystal_ch, crystal_mask)

    if has_momentum:
        return x_padded, mask_padded, indices_padded, torch.stack(momentum_list)
    return x_padded, mask_padded, indices_padded


# Thread-local storage for per-worker HDF5 file handles.
_tls = threading.local()


def _get_h5_handle(path: str) -> "h5py.File":
    """Return a cached, per-thread HDF5 file handle for *path*."""
    if not hasattr(_tls, "handles"):
        _tls.handles = {}
    if path not in _tls.handles:
        _tls.handles[path] = h5py.File(path, "r", swmr=True)
    return _tls.handles[path]


def _build_track_index(
    h5_files: List[str],
    max_tracks: Optional[int],
    cache_path: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """Scan HDF5 files to build a (file_idx, track_name) index (cached)."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("h5_files") == h5_files:
            logging.getLogger(__name__).info(f"Loaded track index from cache: {cache_path}")
            return cached["track_index"]

    track_index: List[Tuple[int, str]] = []
    for file_idx, h5_file in enumerate(h5_files):
        try:
            with h5py.File(h5_file, "r") as f:
                if "tracks" not in f:
                    continue
                for track_name in f["tracks"].keys():
                    track_index.append((file_idx, track_name))
                    if max_tracks and len(track_index) >= max_tracks:
                        break
            if max_tracks and len(track_index) >= max_tracks:
                break
        except Exception:
            pass

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"h5_files": h5_files, "track_index": track_index}, fh)
        logging.getLogger(__name__).info(f"Saved track index cache: {cache_path}")

    return track_index


class MultiFileHDF5Dataset(Dataset):
    """PyTorch Dataset that loads sparse tracks from multiple HDF5 files."""

    def __init__(
        self,
        data_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t0", "t1", "tot0", "tot1"),
        max_tracks: Optional[int] = None,
        index_cache_path: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)
        p = Path(data_path)
        if p.is_dir():
            self.h5_files = [str(x) for x in sorted(p.glob("*.h5"))]
        elif p.is_file():
            self.h5_files = [str(p)]
        else:
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        self.channel_index = channel_index
        self.feature_names = feature_names
        self.track_index = _build_track_index(self.h5_files, max_tracks, index_cache_path)

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file_idx, track_name = self.track_index[idx]
        f = _get_h5_handle(self.h5_files[file_idx])
        return build_sparse_track_from_hdf5_group(
            f["tracks"][track_name], self.channel_index, self.feature_names
        )


class TrackHDF5Dataset(Dataset):
    """PyTorch Dataset for sparse track data stored in a single HDF5 file."""

    def __init__(
        self,
        h5_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t0", "t1", "tot0", "tot1"),
        max_tracks: Optional[int] = None,
    ):
        self.h5_path = str(h5_path)
        self.channel_index = channel_index
        self.feature_names = feature_names

        with h5py.File(self.h5_path, "r") as f:
            self.track_names = list(f["tracks"].keys())
            if max_tracks:
                self.track_names = self.track_names[:max_tracks]

    def __len__(self) -> int:
        return len(self.track_names)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        f = _get_h5_handle(self.h5_path)
        return build_sparse_track_from_hdf5_group(
            f["tracks"][self.track_names[idx]], self.channel_index, self.feature_names
        )


class MomentumTrackDataset(Dataset):
    """PyTorch Dataset for track 3D momentum vector prediction."""

    def __init__(
        self,
        h5_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t0", "t1", "tot0", "tot1"),
        max_tracks: Optional[int] = None,
    ):
        self.h5_path = str(h5_path)
        self.channel_index = channel_index
        self.feature_names = feature_names

        with h5py.File(self.h5_path, "r") as f:
            self.track_names = list(f["tracks"].keys())
            if max_tracks:
                self.track_names = self.track_names[:max_tracks]

    def __len__(self) -> int:
        return len(self.track_names)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        f = _get_h5_handle(self.h5_path)
        track_group = f["tracks"][self.track_names[idx]]
        features, indices = build_sparse_track_from_hdf5_group(
            track_group, self.channel_index, self.feature_names
        )
        px = float(track_group.attrs.get("true_mom_x", 0.0))
        py = float(track_group.attrs.get("true_mom_y", 0.0))
        pz = float(track_group.attrs.get("true_mom_z", 0.0))
        return features, indices, torch.tensor([px, py, pz], dtype=torch.float32)


class FlatHDF5Dataset(Dataset):
    """
    Dataset for the flat CSR-style HDF5 format produced by preprocess_to_flat.py.
    """

    def __init__(self, data_path: str, max_tracks: Optional[int] = None):
        p = Path(data_path)
        if p.is_dir():
            self.h5_files = [str(x) for x in sorted(p.glob("*.h5"))]
        elif p.is_file():
            self.h5_files = [str(p)]
        else:
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        self.track_index: List[Tuple[int, int]] = []
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, "r") as f:
                n = int(f.attrs.get("n_tracks", len(f["offsets"]) - 1))
                for ti in range(n):
                    self.track_index.append((file_idx, ti))
                    if max_tracks and len(self.track_index) >= max_tracks:
                        break
            if max_tracks and len(self.track_index) >= max_tracks:
                break

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file_idx, track_idx = self.track_index[idx]
        f = _get_h5_handle(self.h5_files[file_idx])
        a = int(f["offsets"][track_idx])
        b = int(f["offsets"][track_idx + 1])
        features   = f["features"][a:b]
        ch_indices = f["ch_indices"][a:b]
        return np.asarray(features, dtype=np.float32), np.asarray(ch_indices, dtype=np.int64)


class FlatMomentumDataset(Dataset):
    """
    Flat CSR dataset for momentum prediction.

    tracker-only mode (use_calo=False):
        __getitem__ -> (features, ch_indices, mom)

    tracker + per-crystal calo mode (use_calo=True):
        __getitem__ -> (features, ch_indices, mom,
                        crystal_feats, crystal_ids, n_crystals)

        where the per-crystal block is the ONLY calo information the model
        sees:
          crystal_feats : (n_crys, 2)  float32   [edep, time]
          crystal_ids   : (n_crys,)    int64     raw crystal_id (unshifted;
                                                 the model offsets it before
                                                 the shared PE lookup)
          n_crystals    : int                    length of the above (may be 0
                                                 for tracks that failed the
                                                 track↔cluster match)

        The 22-vector `calo_scalars` is intentionally NOT returned: it
        contained reconstruction outputs (calo_mom_x/y/z, calo_poca_*,
        tresid, etc.) that effectively leaked the target momentum.

    Requires flat files produced by the *updated* preprocess_to_flat.py
    (n_calo_hit_feats == 3, calo_hit_feat_names == ["crystal_id", "edep", "time"]).
    """

    def __init__(
        self,
        data_path: str,
        max_tracks: Optional[int] = None,
        use_calo: bool = False,
    ):
        p = Path(data_path)
        if p.is_dir():
            self.h5_files = [str(x) for x in sorted(p.glob("*.h5"))]
        elif p.is_file():
            self.h5_files = [str(p)]
        else:
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        self.use_calo = use_calo

        # Validate flat-file layout when calo is requested — refuse to run
        # against stale files rather than silently feeding the model 2-feature
        # crystal blocks (no time) and getting mystery losses.
        if use_calo:
            with h5py.File(self.h5_files[0], "r") as _f:
                n_feats = int(_f.attrs.get("n_calo_hit_feats", 0))
                names   = list(_f.attrs.get("calo_hit_feat_names", []))
            if n_feats < 3 or "time" not in names:
                raise RuntimeError(
                    f"--use-calo requested but flat file {self.h5_files[0]} "
                    f"has n_calo_hit_feats={n_feats}, names={names}.  "
                    f"Re-run preprocess_to_flat.py to regenerate the flat "
                    f"files with per-crystal time included "
                    f"(expected: n_calo_hit_feats>=3, names includes 'time')."
                )

        self.track_index: List[Tuple[int, int]] = []
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, "r") as f:
                n = int(f.attrs.get("n_tracks", len(f["offsets"]) - 1))
                for ti in range(n):
                    self.track_index.append((file_idx, ti))
                    if max_tracks and len(self.track_index) >= max_tracks:
                        break
            if max_tracks and len(self.track_index) >= max_tracks:
                break

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int):
        file_idx, track_idx = self.track_index[idx]
        f = _get_h5_handle(self.h5_files[file_idx])
        a = int(f["offsets"][track_idx])
        b = int(f["offsets"][track_idx + 1])
        features   = np.asarray(f["features"][a:b],       dtype=np.float32)
        ch_indices = np.asarray(f["ch_indices"][a:b],     dtype=np.int64)
        mom        = np.asarray(f["mom_xyz"][track_idx],  dtype=np.float32)

        if not self.use_calo:
            return features, ch_indices, torch.from_numpy(mom)

        # --- Tracker + per-crystal calo mode ---
        # Layout in the flat file: (n_tracks, max_crystals, 3) with columns
        #   [crystal_id, edep, time]
        # and n_crystals[track_idx] telling us how many rows are real.
        n_crys = int(f["calo_n_crystals"][track_idx])
        if n_crys > 0:
            padded = np.asarray(f["calo_hits_padded"][track_idx, :n_crys],
                                dtype=np.float32)
            crystal_ids   = padded[:, 0].astype(np.int64)
            crystal_feats = padded[:, 1:3].copy()   # (n_crys, 2) [edep, time]
        else:
            crystal_ids   = np.zeros(0, dtype=np.int64)
            crystal_feats = np.zeros((0, 2), dtype=np.float32)

        return (features, ch_indices, torch.from_numpy(mom),
                crystal_feats, crystal_ids, n_crys)


class MultiFileMomentumDataset(Dataset):
    """PyTorch Dataset for 3D momentum prediction from multiple HDF5 files."""

    def __init__(
        self,
        data_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t0", "t1", "tot0", "tot1"),
        max_tracks: Optional[int] = None,
        index_cache_path: Optional[str] = None,
    ):
        p = Path(data_path)
        if p.is_dir():
            self.h5_files = [str(x) for x in sorted(p.glob("*.h5"))]
        else:
            self.h5_files = [str(p)]

        self.channel_index = channel_index
        self.feature_names = feature_names
        self.track_index = _build_track_index(self.h5_files, max_tracks, index_cache_path)

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        file_idx, track_name = self.track_index[idx]
        f = _get_h5_handle(self.h5_files[file_idx])
        track_group = f["tracks"][track_name]
        features, indices = build_sparse_track_from_hdf5_group(
            track_group, self.channel_index, self.feature_names
        )
        px = float(track_group.attrs.get("true_mom_x", 0.0))
        py = float(track_group.attrs.get("true_mom_y", 0.0))
        pz = float(track_group.attrs.get("true_mom_z", 0.0))
        return features, indices, torch.tensor([px, py, pz], dtype=torch.float32)


def create_dataloaders(
    train_data_path: str,
    val_data_path: str,
    channel_index: Dict[str, int],
    batch_size: int = 32,
    num_workers: int = 0,
    feature_names: Tuple[str, ...] = ("t0", "t1", "tot0", "tot1"),
    task: str = "reconstruction",
    max_tracks: Optional[int] = None,
    output_dir: Optional[str] = None,
    flat_format: bool = False,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    momentum_like = task in ("momentum", "abs_momentum", "cvn_momentum")

    if flat_format:
        if momentum_like:
            use_calo = kwargs.get("use_calo", False)
            train_dataset = FlatMomentumDataset(train_data_path, max_tracks, use_calo=use_calo)
            val_dataset   = FlatMomentumDataset(val_data_path,   max_tracks, use_calo=use_calo)
        else:
            train_dataset = FlatHDF5Dataset(train_data_path, max_tracks)
            val_dataset   = FlatHDF5Dataset(val_data_path,   max_tracks)
    else:
        train_is_dir = Path(train_data_path).is_dir()
        val_is_dir   = Path(val_data_path).is_dir()

        train_cache = str(Path(output_dir) / "train_index.pkl") if output_dir else None
        val_cache   = str(Path(output_dir) / "val_index.pkl")   if output_dir else None

        if momentum_like:
            train_dataset = (MultiFileMomentumDataset(train_data_path, channel_index, feature_names, max_tracks, train_cache)
                             if train_is_dir else MomentumTrackDataset(train_data_path, channel_index, feature_names, max_tracks))
            val_dataset   = (MultiFileMomentumDataset(val_data_path, channel_index, feature_names, max_tracks, val_cache)
                             if val_is_dir else MomentumTrackDataset(val_data_path, channel_index, feature_names, max_tracks))
        else:
            train_dataset = (MultiFileHDF5Dataset(train_data_path, channel_index, feature_names, max_tracks, train_cache)
                             if train_is_dir else TrackHDF5Dataset(train_data_path, channel_index, feature_names, max_tracks))
            val_dataset   = (MultiFileHDF5Dataset(val_data_path, channel_index, feature_names, max_tracks, val_cache)
                             if val_is_dir else TrackHDF5Dataset(val_data_path, channel_index, feature_names, max_tracks))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=sparse_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=sparse_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Batch-unpacking helper shared by train_epoch and validate.
# The batch tuple length tells us which mode we're in:
#   3  : reconstruction / denoising          (x, mask, ch)
#   4  : tracker-only momentum               (x, mask, ch, target)
#   7  : tracker + per-crystal calo momentum (x, mask, ch, target,
#                                             crystal_x, crystal_ch, crystal_mask)
# ---------------------------------------------------------------------------
def _unpack_momentum_batch(
    batch_data: Tuple, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if len(batch_data) == 7:
        (x, mask, ch, target,
         crystal_x, crystal_ch, crystal_mask) = batch_data
        crystal_x    = crystal_x.to(device, non_blocking=True)
        crystal_ch   = crystal_ch.to(device, non_blocking=True)
        crystal_mask = crystal_mask.to(device, non_blocking=True)
    else:
        x, mask, ch, target = batch_data
        crystal_x = crystal_ch = crystal_mask = None

    x      = x.to(device, non_blocking=True)
    mask   = mask.to(device, non_blocking=True)
    ch     = ch.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    return x, mask, ch, target, crystal_x, crystal_ch, crystal_mask


class DivergenceError(RuntimeError):
    """Raised when an epoch exceeds the allowed non-finite-batch fraction.

    Raising this from ``train_epoch`` is the signal to ``main`` that the run
    has entered the frozen-model regime (see the epoch-11 pathology in
    runs/cvn_momentum_v1/logs/train_20260814_153907.log) and should bail out
    cleanly rather than grinding through the remaining epochs producing an
    identical, useless loss every time.
    """


def _tensor_finite_summary(name: str, value: Optional[torch.Tensor]) -> str:
    """Return a compact finite-value/range summary for failure logging."""
    if value is None:
        return f"{name}=None"
    detached = value.detach()
    finite = torch.isfinite(detached)
    n_bad = int((~finite).sum().item())
    if finite.any():
        finite_values = detached[finite].float()
        value_range = (
            f"range=[{finite_values.min().item():.4g},"
            f"{finite_values.max().item():.4g}]"
        )
    else:
        value_range = "range=[no finite values]"
    return (
        f"{name}:shape={tuple(detached.shape)},dtype={detached.dtype},"
        f"nonfinite={n_bad},{value_range}"
    )


def _model_parameter_summary(model: nn.Module) -> Tuple[str, float, List[str]]:
    """Find the largest parameter and all currently non-finite parameters."""
    max_name = "<none>"
    max_value = 0.0
    nonfinite_names: List[str] = []
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        detached = parameter.detach()
        if not torch.isfinite(detached).all():
            nonfinite_names.append(name)
            continue
        value = detached.abs().max().item()
        if value > max_value:
            max_name, max_value = name, value
    return max_name, max_value, nonfinite_names


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    loss_fn: callable = masked_mse_loss,
    accumulation_steps: int = 1,
    task: str = "reconstruction",
    epoch_number: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    batch_lr_update: Optional[callable] = None,
    grad_clip: Optional[float] = 1.0,
    divergence_frac: Optional[float] = 0.20,
    divergence_min_batches: int = 1000,
    weight_probe_every: Optional[int] = 10_000,
) -> float:
    """Train for one epoch with AMP.

    Parameters
    ----------
    divergence_frac
        If the ratio ``nan_batches / batches_seen`` exceeds this value once at
        least ``divergence_min_batches`` batches have been seen, raise
        :class:`DivergenceError`.  Set to ``None`` to disable the check.
    divergence_min_batches
        Minimum number of batches to observe before the divergence-fraction
        check is armed.  Prevents a single fp16 hiccup at step 3 from taking
        the whole run down.
    weight_probe_every
        Every N batches, log the name and magnitude of the largest parameter,
        current GradScaler scale, and parameter finite status. Set to ``None``
        to disable.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = device.type == "cuda"
    amp_device = device.type

    pbar = tqdm(dataloader, desc="Training", total=steps_per_epoch)
    momentum_like = task in ("momentum", "abs_momentum", "cvn_momentum")
    logger = logging.getLogger(__name__)
    nan_batches = 0
    batches_seen = 0  # counts EVERY batch we tried, including skips
    for batch_idx, batch_data in enumerate(pbar):
        if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
            break
        if momentum_like:
            (x, mask, ch, target,
             crystal_x, crystal_ch, crystal_mask) = _unpack_momentum_batch(batch_data, device)
            with autocast(amp_device, enabled=use_amp):
                output = model(
                    x, mask=mask, channel_indices=ch,
                    crystal_x=crystal_x,
                    crystal_ch=crystal_ch,
                    crystal_mask=crystal_mask,
                )
                loss = loss_fn(output, target) / accumulation_steps
        else:
            x, mask, ch = batch_data
            x    = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            ch   = ch.to(device, non_blocking=True)
            with autocast(amp_device, enabled=use_amp):
                output = model(x, mask=mask, channel_indices=ch)
                loss = loss_fn(output, x, mask) / accumulation_steps

        # ----------------------------------------------------------
        # NaN/Inf safety valve.  If a single batch's forward produced
        # a non-finite loss we MUST NOT call .backward() on it --
        # doing so poisons the parameter gradients with NaN and, once
        # the optimizer steps, every subsequent forward pass returns
        # NaN as well (visible as the tqdm postfix locking to
        # 'loss=nan' for the rest of the epoch).  Instead we:
        #   * skip backward + optimizer step for this batch,
        #   * zero any partially-accumulated grads from an in-flight
        #     accumulation window,
        #   * still advance the LR schedule so it stays aligned with
        #     the global step count reported by tqdm,
        #   * log a warning (throttled) so this is visible upstream.
        # ----------------------------------------------------------
        batches_seen += 1

        if not torch.isfinite(loss):
            nan_batches += 1
            if nan_batches <= 5 or nan_batches % 100 == 0:
                max_name, max_value, bad_params = _model_parameter_summary(model)
                logger.warning(
                    f"Non-finite loss at epoch {epoch_number}, batch {batch_idx} "
                    f"(loss.item()={loss.item()!r}); skipping backward/step. "
                    f"Total skipped this epoch: {nan_batches}. "
                    f"{_tensor_finite_summary('x', x)}; "
                    f"{_tensor_finite_summary('target', target if momentum_like else x)}; "
                    f"{_tensor_finite_summary('output', output)}; "
                    f"largest_param={max_name}:{max_value:.4g}; "
                    f"nonfinite_params={bad_params or 'none'}"
                )
            optimizer.zero_grad(set_to_none=True)
            if (batch_idx + 1) % accumulation_steps == 0 and batch_lr_update is not None:
                batch_lr_update()
            pbar.set_postfix({
                "loss": f"{(total_loss / max(num_batches, 1)):.4f}",
                "nan":  nan_batches,
            })
            # Arm the divergence guard even on loss-side NaN — a stream of
            # NaN losses is exactly the frozen-model regime we want to catch.
            if (divergence_frac is not None
                    and batches_seen >= divergence_min_batches
                    and nan_batches / batches_seen > divergence_frac):
                raise DivergenceError(
                    f"{nan_batches}/{batches_seen} batches ("
                    f"{100.0 * nan_batches / batches_seen:.1f}%) produced "
                    f"non-finite loss/grad this epoch — exceeds "
                    f"{100.0 * divergence_frac:.1f}% threshold. "
                    f"Model has diverged; aborting."
                )
            continue

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Always unscale and inspect before zero_grad, even when clipping
            # is disabled. GradScaler.step() would otherwise silently skip an
            # overflow without identifying the tensor that caused it.
            scaler.unscale_(optimizer)
            bad_gradients = [
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if not bad_gradients:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()

                # Finite gradients should not produce non-finite parameters.
                # Detect optimizer/state corruption before gradients are
                # cleared or the epoch can be checkpointed as healthy.
                _, _, bad_params = _model_parameter_summary(model)
                if bad_params:
                    raise DivergenceError(
                        f"Optimizer step produced non-finite parameters at "
                        f"epoch {epoch_number}, batch {batch_idx}: "
                        f"{bad_params[:12]}"
                        f"{'...' if len(bad_params) > 12 else ''}."
                    )
            else:
                nan_batches += 1
                if nan_batches <= 5 or nan_batches % 100 == 0:
                    max_name, max_value, bad_params = _model_parameter_summary(model)
                    grad_summaries = [
                        _tensor_finite_summary(f"grad[{name}]", gradient)
                        for name, gradient in bad_gradients[:12]
                    ]
                    logger.warning(
                        f"Non-finite gradient at epoch {epoch_number}, "
                        f"batch {batch_idx}; scaler_scale="
                        f"{scaler.get_scale():.4g}; "
                        f"offending_gradients={grad_summaries}"
                        f"{'...' if len(bad_gradients) > 12 else ''}; "
                        f"skipping optimizer step. Total skipped this "
                        f"epoch: {nan_batches}. largest_param="
                        f"{max_name}:{max_value:.4g}; "
                        f"nonfinite_params={bad_params or 'none'}"
                    )
                # Decrement the scale after the skipped optimizer step.
                scaler.update()
                if (divergence_frac is not None
                        and batches_seen >= divergence_min_batches
                        and nan_batches / batches_seen > divergence_frac):
                    raise DivergenceError(
                        f"{nan_batches}/{batches_seen} batches ("
                        f"{100.0 * nan_batches / batches_seen:.1f}%) "
                        f"produced non-finite loss/grad this epoch — "
                        f"exceeds {100.0 * divergence_frac:.1f}% "
                        f"threshold. Model has diverged; aborting."
                    )
            optimizer.zero_grad(set_to_none=True)
            if batch_lr_update is not None:
                batch_lr_update()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        # ------------------------------------------------------------
        # Weight/grad magnitude probe.  A run entering the pathological
        # "large-but-finite weights → fp16 grad overflow → frozen model"
        # regime (see runs/cvn_momentum_v1/logs/train_20260814_153907.log,
        # epochs 8→9) shows max|param| climbing sharply BEFORE the
        # nan_batches fraction blows up.  Logging these lets us catch
        # the problem an epoch or two earlier than the divergence check.
        # ------------------------------------------------------------
        if (weight_probe_every is not None
                and weight_probe_every > 0
                and batches_seen % weight_probe_every == 0):
            max_name, max_p, bad_params = _model_parameter_summary(model)
            logger.info(
                f"[probe] batch {batches_seen:,}: "
                f"max|param|={max_p:.3g} ({max_name})  "
                f"nonfinite_params={bad_params or 'none'}  "
                f"scaler_scale={scaler.get_scale():.3g}  "
                f"nan_batches={nan_batches}"
            )

    if nan_batches > 0:
        logger.warning(
            f"Epoch had {nan_batches} skipped batch(es) due to non-finite "
            f"loss or gradients."
        )

    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: callable = masked_mse_loss,
    task: str = "reconstruction",
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    use_amp = device.type == "cuda"
    amp_device = device.type

    momentum_like = task in ("momentum", "abs_momentum", "cvn_momentum")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_data in pbar:
            if momentum_like:
                (x, mask, ch, target,
                 crystal_x, crystal_ch, crystal_mask) = _unpack_momentum_batch(batch_data, device)
                with autocast(amp_device, enabled=use_amp):
                    output = model(
                        x, mask=mask, channel_indices=ch,
                        crystal_x=crystal_x,
                        crystal_ch=crystal_ch,
                        crystal_mask=crystal_mask,
                    )
                    loss = loss_fn(output, target)
            else:
                x, mask, ch = batch_data
                x    = x.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                ch   = ch.to(device, non_blocking=True)
                with autocast(amp_device, enabled=use_amp):
                    output = model(x, mask=mask, channel_indices=ch)
                    loss = loss_fn(output, x, mask)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return total_loss / num_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sparse track transformer model.")
    parser.add_argument("train_data", help="Path to training HDF5 file or directory.")
    parser.add_argument("val_data", help="Path to validation HDF5 file or directory.")
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum global gradient norm after AMP unscale. "
                             "Set to <=0 to disable clipping.")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of linear-warmup epochs at the start of "
                             "training.  Longer warmup flattens the LR ramp "
                             "so an outlier gradient at peak LR is less "
                             "likely to knock the model into the frozen-"
                             "weights divergence regime; see epoch-9 event "
                             "in runs/cvn_momentum_v1/logs/train_20260814_"
                             "153907.log.  Default was 2 for the tracker-"
                             "only baseline.")
    parser.add_argument("--keep-last-n", type=int, default=3,
                        help="Save a rolling checkpoint every epoch and keep "
                             "the last N.  Complements best_model.pt (which "
                             "only tracks best-val-loss) by giving you a "
                             "known-good state to restart from when a run "
                             "diverges mid-training.  Set to 0 to disable.")
    parser.add_argument("--divergence-frac", type=float, default=0.20,
                        help="Fraction of non-finite loss/grad batches per "
                             "epoch that triggers a clean abort. Prevents "
                             "grinding through days of frozen-model epochs. "
                             "Set to <=0 to disable.")
    parser.add_argument("--max-val-loss-ratio", type=float, default=10.0,
                        help="Abort without writing a rolling checkpoint if "
                             "validation loss exceeds this multiple of the "
                             "previous best. Set to <=0 to disable.")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Default max_channels bumped so the shared positional-encoding table has
    # room for BOTH straws (0..41471) and crystals (CRYSTAL_PE_OFFSET..).
    parser.add_argument("--max-channels", type=int, default=50000)
    parser.add_argument(
        "--model-type",
        choices=["reconstruction", "denoising", "momentum", "abs_momentum", "cvn_momentum"],
        default="reconstruction",
        help="Task head to train. 'momentum' predicts (px, py, pz); "
             "'abs_momentum' predicts (pT, |pz|) to break the forward/backward "
             "magnetic-mirror degeneracy in the CE isotropic MC; "
             "'cvn_momentum' uses a geometry-aware convolutional-neighbor "
             "stem in front of the transformer and predicts (pT, |pz|) "
             "(same degeneracy-safe target as abs_momentum).",
    )
    parser.add_argument("--loss-fn", choices=["mse", "l1"], default="mse")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--infer-channel-index", type=int, default=None)
    parser.add_argument("--pooling-type", choices=["mean", "max", "attention"], default="mean")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--flat-format", action="store_true", default=False)
    parser.add_argument("--use-calo", action="store_true", default=False,
                        help="Feed per-crystal calorimeter hits (edep, time; "
                             "crystal_id via positional encoding) as extra "
                             "tokens into the same transformer as the straw "
                             "hits.  Requires flat HDF5 files produced by the "
                             "updated preprocess_to_flat.py (n_calo_hit_feats=3, "
                             "including per-crystal time).  Only applies with "
                             "--flat-format and --model-type in "
                             "{momentum, abs_momentum, cvn_momentum}.")
    parser.add_argument("--allow-resume-config-mismatch", action="store_true",
                        help="Explicitly allow a resume whose saved optimizer/LR "
                             "schedule settings differ from this invocation. "
                             "This makes the continuation non-bit-exact and can "
                             "destabilize training.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume training from a checkpoint in "
                             "--output-dir.  Pass a path to a specific .pt "
                             "file, or use --resume with no argument (or "
                             "--resume auto) to pick the highest-numbered "
                             "epoch_NNN.pt in the run directory, falling "
                             "back to best_model.pt if no rolling checkpoints "
                             "exist.  Rolling checkpoints (epoch_NNN.pt) "
                             "carry optimizer + scaler state and give a "
                             "bit-exact resume; best_model.pt only carries "
                             "the model weights, so resuming from it will "
                             "restart the optimizer/scaler from scratch "
                             "(the run will still make progress, but the "
                             "first epoch after resume may spike briefly).")

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs")

    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Resolve the --resume argument up front (before model/optimizer
    # construction) so we can (a) log which checkpoint we're picking up
    # from and (b) fail fast if the requested checkpoint doesn't exist.
    # Actual state loading happens after model/optimizer/scaler are
    # created below.
    # ------------------------------------------------------------------
    resume_ckpt_path: Optional[Path] = None
    if args.resume is not None:
        if args.resume == "auto":
            # Prefer the highest-numbered rolling checkpoint (has optim
            # + scaler state), else fall back to best_model.pt.
            rolling = sorted(output_dir.glob("epoch_*.pt"))
            if rolling:
                resume_ckpt_path = rolling[-1]
                logger.info(
                    f"--resume auto: picked {resume_ckpt_path.name} "
                    f"(latest of {len(rolling)} rolling checkpoint(s))."
                )
            elif (output_dir / "best_model.pt").exists():
                resume_ckpt_path = output_dir / "best_model.pt"
                logger.warning(
                    "--resume auto: no epoch_NNN.pt found, falling back "
                    "to best_model.pt.  This has no optimizer/scaler "
                    "state, so training will restart those from scratch."
                )
            else:
                raise FileNotFoundError(
                    f"--resume requested but no checkpoints found under "
                    f"{output_dir!r} (looked for epoch_*.pt and "
                    f"best_model.pt)."
                )
        else:
            resume_ckpt_path = Path(args.resume)
            if not resume_ckpt_path.is_file():
                raise FileNotFoundError(
                    f"--resume path {resume_ckpt_path!r} does not exist."
                )
            logger.info(f"--resume: loading from {resume_ckpt_path}.")

    train_data_resolved = _resolve_data_paths(args.train_data)
    val_data_resolved   = _resolve_data_paths(args.val_data)

    logger.info("Building channel index...")
    if args.infer_channel_index:
        channel_index = infer_channel_index_from_hdf5(train_data_resolved, max_tracks=args.infer_channel_index)
    else:
        channel_index = {
            f"{p}_{pa}_{l}_{s}": idx
            for idx, (p, pa, l, s) in enumerate(
                [(p, pa, l, s) for p in range(36) for pa in range(6) for l in range(2) for s in range(96)]
            )
        }

    use_calo = (
        args.use_calo
        and args.flat_format
        and args.model_type in ("momentum", "abs_momentum", "cvn_momentum")
    )
    if args.use_calo and not use_calo:
        logger.warning("--use-calo requires --flat-format and --model-type in "
                       "{momentum, abs_momentum, cvn_momentum}; ignoring.")

    # Sanity-check the max_channels vs the PE offset that the model applies
    # internally to crystal IDs.  Without enough headroom straw and crystal
    # PE rows would collide (they'd still be disambiguated by the learned
    # detector-type embedding, but you lose positional cleanliness).
    if use_calo and args.max_channels < CRYSTAL_PE_OFFSET + 5000:
        logger.warning(
            f"--max-channels={args.max_channels} is small for --use-calo "
            f"(CRYSTAL_PE_OFFSET={CRYSTAL_PE_OFFSET}); "
            f"consider --max-channels {CRYSTAL_PE_OFFSET + 5000} or larger."
        )

    if use_calo:
        logger.info(
            "--use-calo enabled: per-crystal (edep, time) hits are fused as "
            "extra tokens into the transformer, with a learned detector-type "
            "embedding to distinguish straws from crystals and the shared PE "
            f"table offset by CRYSTAL_PE_OFFSET={CRYSTAL_PE_OFFSET} for crystals."
        )

    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data_resolved, val_data_resolved, channel_index,
        batch_size=args.batch_size, num_workers=args.num_workers, task=args.model_type,
        max_tracks=args.max_tracks, output_dir=str(output_dir), flat_format=args.flat_format,
        use_calo=use_calo,
    )
    logger.info(f"Train tracks: {len(train_loader.dataset):,} | Val tracks: {len(val_loader.dataset):,}")
    logger.info(f"Train batches/epoch: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # ------------------------------------------------------------------
    # Auto-detect the tracker per-hit feature width from the actual data.
    #
    # The flat-format files carry an ``n_features`` attribute written by
    # preprocess_to_flat.py; when running against the older 3-feature files
    # (t0, t1, edep) this will be 3, while the new 4-feature files
    # (t0, t1, tot0, tot1) report 4.  For the legacy per-track HDF5 path
    # we fall back to peeking at one sample from the dataset.
    # ------------------------------------------------------------------
    input_dim = None
    if args.flat_format:
        first_flat = train_loader.dataset.h5_files[0]
        with h5py.File(first_flat, "r") as _f:
            n_feat = int(_f.attrs.get("n_features", 0))
            names  = list(_f.attrs.get("feature_names", []))
        if n_feat > 0:
            input_dim = n_feat
            logger.info(
                f"Auto-detected input_dim={input_dim} from flat file "
                f"(feature_names={names})"
            )
    if input_dim is None:
        # Peek at one sample from the train dataset.  Works for all
        # dataset variants because they all return (features, ...) with
        # features shape (n_hits, n_feat).
        sample = train_loader.dataset[0]
        input_dim = int(sample[0].shape[1])
        logger.info(f"Auto-detected input_dim={input_dim} from sample track 0")

    config = TrackModelConfig(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, max_channels=args.max_channels,
        task=args.model_type, pooling_type=args.pooling_type,
        use_calo=use_calo,
    )

    if args.model_type == "denoising":
        model = DenoisingTrackModel(**config.to_dict())
    else:
        model = TrackReconstructionModel(**config.to_dict())

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)

    if args.compile:
        logger.info("Applying torch.compile(dynamic=True) to model...")
        model = torch.compile(model, dynamic=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    # Persist every setting that affects optimizer updates or the manual LR
    # schedule. Architecture/task settings remain in ``config``.
    training_config: Dict[str, Any] = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "warmup_epochs": args.warmup_epochs,
        "epochs": args.epochs,
        "train_batches_per_epoch": len(train_loader),
        "steps_per_epoch": args.steps_per_epoch,
        "accumulation_steps": args.accumulation_steps,
        "batch_size": args.batch_size,
        "min_lr": 1e-6,
        "warmup_start_factor": 0.1,
    }

    # ------------------------------------------------------------------
    # Load resume checkpoint (if any) *after* model/optimizer/scaler
    # exist but *before* the LR schedule is stepped, so global_step and
    # the LR-at-resume line up.
    # ------------------------------------------------------------------
    resume_epoch = 0          # number of epochs already completed
    resume_global_step = 0    # optimizer.step() count already done
    resume_best_val = float("inf")
    resume_history: Optional[Dict[str, list]] = None
    resume_had_optim_state = False

    if resume_ckpt_path is not None:
        # weights_only=False because our checkpoints legitimately store
        # more than tensors (config dict, epoch, etc.).  Only load
        # checkpoints you produced yourself.
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)

        ckpt_config = ckpt.get("config")
        if ckpt_config is not None:
            # Cross-check the fields that would silently corrupt a
            # resume if they mismatched.  d_model/nhead/num_layers
            # differences would fail loudly on state_dict load anyway;
            # task/input_dim/use_calo are subtler and worth flagging.
            for key in ("task", "input_dim", "use_calo", "d_model",
                        "nhead", "num_layers", "max_channels"):
                new_val = config.to_dict().get(key)
                old_val = ckpt_config.get(key)
                if old_val is not None and old_val != new_val:
                    raise ValueError(
                        f"--resume: checkpoint {resume_ckpt_path.name} was "
                        f"trained with {key}={old_val!r} but this run is "
                        f"configured for {key}={new_val!r}.  Refusing to "
                        f"resume with a mismatched architecture / task."
                    )

        saved_training_config = ckpt.get("training_config")
        if saved_training_config is None:
            logger.warning(
                f"Checkpoint {resume_ckpt_path.name} has no training_config; "
                "LR/schedule compatibility cannot be validated."
            )
        else:
            mismatches = {
                key: (saved_training_config.get(key), value)
                for key, value in training_config.items()
                if saved_training_config.get(key) != value
            }
            if mismatches:
                message = "; ".join(
                    f"{key}: checkpoint={old!r}, requested={new!r}"
                    for key, (old, new) in mismatches.items()
                )
                if not args.allow_resume_config_mismatch:
                    raise ValueError(
                        "--resume training configuration mismatch: " + message
                        + ". Pass --allow-resume-config-mismatch to override."
                    )
                logger.warning("Explicit resume config override: " + message)

        # Strip a leading '_orig_mod.' if the saved model was compiled
        # (torch.compile prepends that prefix).  Same when loading a
        # non-compiled ckpt into a compiled model.
        state_dict = ckpt["model_state_dict"]
        cur_compiled = any(k.startswith("_orig_mod.") for k in model.state_dict())
        ckpt_compiled = any(k.startswith("_orig_mod.") for k in state_dict)
        if cur_compiled and not ckpt_compiled:
            state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        elif ckpt_compiled and not cur_compiled:
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resume_had_optim_state = True
        if "scaler_state_dict" in ckpt and scaler.is_enabled():
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not restore GradScaler state: {e}")

        resume_epoch = int(ckpt.get("epoch", 0))
        resume_global_step = int(ckpt.get("global_step", 0))
        resume_best_val = float(ckpt.get("val_loss", float("inf")))

        # Reload history.json if it exists, trimming to the resumed epoch
        # so we don't duplicate future epochs on top of already-recorded ones.
        hist_path = output_dir / "history.json"
        if hist_path.exists():
            try:
                with open(hist_path, "r") as f:
                    resume_history = json.load(f)
                # Trim to at most resume_epoch entries (checkpoint was
                # saved AFTER that epoch's val_loss was appended).
                for k in ("train_loss", "val_loss"):
                    if k in resume_history:
                        resume_history[k] = resume_history[k][:resume_epoch]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not reload history.json: {e}")
                resume_history = None

    # ------------------------------------------------------------------
    # Manual LR schedule (unchanged from the tracker-only regime).
    # ------------------------------------------------------------------
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    base_lr = args.lr
    min_lr = 1e-6
    warmup_start_factor = 0.1

    def compute_lr(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            frac = step / max(1, warmup_steps)
            return base_lr * (warmup_start_factor + (1.0 - warmup_start_factor) * frac)
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / denom)
        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def set_lr(step: int) -> float:
        lr = compute_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    # If resuming, step the LR schedule to the correct point BEFORE the
    # first optimizer step of the resumed run.  Otherwise the first
    # batch after resume would run at warmup-start LR and stomp on the
    # cosine-decayed state.
    set_lr(resume_global_step)

    best_val_loss = float("inf")
    patience, patience_counter = 15, 0
    history = {"train_loss": [], "val_loss": []}
    rolling_ckpts: List[Path] = []  # FIFO of last-N per-epoch checkpoints
    divergence_frac = args.divergence_frac if args.divergence_frac > 0 else None

    # ------------------------------------------------------------------
    # Apply the resume state on top of the fresh defaults above.
    # ------------------------------------------------------------------
    start_epoch = 0
    if resume_ckpt_path is not None:
        start_epoch = resume_epoch
        if resume_history is not None:
            # Preserve the caller-supplied keys but overwrite the two we
            # track; anything else already in the file (e.g. custom
            # metrics added by future code) is left alone.
            history.update(resume_history)
            best_val_loss = min(
                (v for v in history.get("val_loss", []) if math.isfinite(v)),
                default=float("inf"),
            )
        else:
            best_val_loss = resume_best_val

        # Recompute patience_counter from history: count trailing epochs
        # that did NOT improve on best_val_loss.  This mirrors what the
        # original run would have had at this point in time.
        val_losses = history.get("val_loss", [])
        if val_losses and math.isfinite(best_val_loss):
            trailing = 0
            for v in reversed(val_losses):
                if math.isfinite(v) and v <= best_val_loss:
                    break
                trailing += 1
            patience_counter = trailing
        # Repopulate the rolling-checkpoint FIFO from what's on disk so
        # eviction picks up where the previous run left off.
        rolling_ckpts = sorted(output_dir.glob("epoch_*.pt"))

        logger.info(
            f"Resumed from {resume_ckpt_path.name}: start_epoch={start_epoch + 1}, "
            f"global_step={resume_global_step:,}, best_val_loss={best_val_loss:.4f}, "
            f"patience_counter={patience_counter}, "
            f"optim_state_restored={resume_had_optim_state}, "
            f"history_epochs_loaded={len(history.get('val_loss', []))}"
        )

    if args.model_type == "momentum":
        loss_fn = momentum_loss
    elif args.model_type in ("abs_momentum", "cvn_momentum"):
        # cvn_momentum uses the same (pT, |pz|) target as abs_momentum so
        # it never has to fit through the forward/backward or φ-rotational
        # degeneracies of the tracker.
        loss_fn = abs_momentum_loss
    else:
        loss_fn = masked_mse_loss if args.loss_fn == "mse" else masked_l1_loss

    logger.info("Starting training...")
    logger.info(
        f"LR schedule: linear warmup {warmup_start_factor:g}*lr → {base_lr:g} over "
        f"{warmup_steps:,} steps, then cosine decay to {min_lr:g} over the remaining "
        f"{max(0, total_steps - warmup_steps):,} steps."
    )

    global_step = resume_global_step
    if start_epoch >= args.epochs:
        logger.warning(
            f"start_epoch={start_epoch} >= --epochs={args.epochs}; "
            f"nothing to do.  Pass a larger --epochs to continue training."
        )
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        def _lr_updater(step_delta: int = 1) -> None:
            nonlocal global_step
            global_step += step_delta
            set_lr(global_step)

        try:
            train_loss = train_epoch(
                model, train_loader, optimizer, device, scaler,
                loss_fn=loss_fn, accumulation_steps=args.accumulation_steps,
                task=args.model_type, epoch_number=epoch + 1,
                steps_per_epoch=args.steps_per_epoch,
                batch_lr_update=_lr_updater,
                grad_clip=args.grad_clip if args.grad_clip > 0 else None,
                divergence_frac=divergence_frac,
            )
        except DivergenceError as exc:
            logger.error(
                f"Training aborted at epoch {epoch + 1}: {exc}  "
                f"Last-known-good rolling checkpoints (if any): "
                f"{[p.name for p in rolling_ckpts]}.  "
                f"best_model.pt still reflects the best-val epoch prior to divergence.  "
                f"Recommended next run: reduce --lr (e.g. halve it), increase "
                f"--warmup-epochs, and/or lower --grad-clip; then resume from "
                f"one of the rolling checkpoints."
            )
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            return
        val_loss = validate(model, val_loader, device, loss_fn=loss_fn, task=args.model_type)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | step: {global_step:,}"
        )

        val_diverged = (
            not math.isfinite(val_loss)
            or (args.max_val_loss_ratio > 0
                and math.isfinite(best_val_loss)
                and val_loss > args.max_val_loss_ratio * best_val_loss)
        )
        if val_diverged:
            logger.error(
                f"Validation diverged at epoch {epoch + 1}: val_loss={val_loss!r}, "
                f"previous_best={best_val_loss!r}; refusing to checkpoint."
            )
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "training_config": training_config,
                },
                output_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        # ------------------------------------------------------------
        # Rolling last-N per-epoch checkpoint.  Kept separately from
        # best_model.pt so that if the run diverges we always have a
        # known-good state from N epochs back to restart from.
        # ------------------------------------------------------------
        if args.keep_last_n > 0:
            ckpt_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config.to_dict(),
                    "training_config": training_config,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            rolling_ckpts.append(ckpt_path)
            # Evict oldest checkpoints beyond keep_last_n.
            while len(rolling_ckpts) > args.keep_last_n:
                stale = rolling_ckpts.pop(0)
                try:
                    stale.unlink()
                except OSError:
                    logger.warning(f"Could not delete stale checkpoint {stale}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()