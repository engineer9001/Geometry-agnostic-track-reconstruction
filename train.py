import argparse
import json
import logging
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
import h5py
from tqdm import tqdm

from model import (
    TrackReconstructionModel,
    DenoisingTrackModel,
    TrackModelConfig,
    masked_mse_loss,
    masked_l1_loss,
    momentum_loss,
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


def sparse_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Stitches variable-length sparse tracks into tightly packaged batch tensors.
    Pads only up to the longest sequence *inside this specific batch*.

    Supported tuple lengths:
        2: (features, indices)                          — reconstruction
        3: (features, indices, momentum)                — momentum, tracker-only
        4: (features, indices, momentum, calo_scalars)  — momentum, tracker+calo
    """
    n_items = len(batch[0])
    has_momentum   = n_items >= 3
    has_calo       = n_items == 4

    if has_calo:
        features_list, indices_list, momentum_list, calo_list = zip(*batch)
    elif has_momentum:
        features_list, indices_list, momentum_list = zip(*batch)
    else:
        features_list, indices_list = zip(*batch)

    batch_size = len(batch)
    max_hits = max(f.shape[0] for f in features_list)
    if max_hits == 0:
        max_hits = 1

    n_features = features_list[0].shape[1]

    # Allocate tightly budgeted tensors
    x_padded = torch.zeros((batch_size, max_hits, n_features), dtype=torch.float32)
    mask_padded = torch.ones((batch_size, max_hits), dtype=torch.bool)
    indices_padded = torch.zeros((batch_size, max_hits), dtype=torch.long)

    for i in range(batch_size):
        n_hits = features_list[i].shape[0]
        if n_hits > 0:
            x_padded[i, :n_hits] = torch.from_numpy(features_list[i])
            mask_padded[i, :n_hits] = False
            indices_padded[i, :n_hits] = torch.from_numpy(indices_list[i])

    if has_calo:
        return x_padded, mask_padded, indices_padded, torch.stack(momentum_list), torch.stack(calo_list)
    if has_momentum:
        return x_padded, mask_padded, indices_padded, torch.stack(momentum_list)
    return x_padded, mask_padded, indices_padded


# Thread-local storage for per-worker HDF5 file handles.
# Each DataLoader worker gets its own open handle, eliminating the
# open/close overhead that would otherwise occur on every __getitem__ call.
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
    """
    Scan HDF5 files to build a (file_idx, track_name) index.
    If cache_path is given, the result is saved/loaded from disk so that
    subsequent runs skip the expensive scan entirely.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        # Validate the cache is still for the same set of files
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
        feature_names: Tuple[str, ...] = ("t_diff", "edep"),
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
        feature_names: Tuple[str, ...] = ("t_diff", "edep"),
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
        feature_names: Tuple[str, ...] = ("t_diff", "edep"),
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
    Each __getitem__ is two numpy array slices — no string decoding, no dict lookup,
    no HDF5 group traversal.
    """

    def __init__(self, data_path: str, max_tracks: Optional[int] = None):
        p = Path(data_path)
        if p.is_dir():
            self.h5_files = [str(x) for x in sorted(p.glob("*.h5"))]
        elif p.is_file():
            self.h5_files = [str(p)]
        else:
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        # Build a flat track index: (file_idx, track_idx_within_file)
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
        features   = f["features"][a:b]    # (n_hits, n_features)
        ch_indices = f["ch_indices"][a:b]  # (n_hits,)
        return np.asarray(features, dtype=np.float32), np.asarray(ch_indices, dtype=np.int64)


class FlatMomentumDataset(Dataset):
    """
    Flat CSR dataset for momentum prediction.
    Each __getitem__ is two array slices + one row read from mom_xyz.

    Args:
        pz_min:   If set, only tracks with true_pz > pz_min are included.
                  Use pz_min=0.0 to restrict to forward-going tracks only.
        use_calo: If True, also return calo_scalars as a 4th element.
                  NaN values (unmatched tracks) are replaced with 0 and a
                  calo_matched flag is appended as the last element, so the
                  model can learn to ignore calo when no cluster was matched.
                  Requires flat files produced with preprocess_to_flat.py
                  from HDF5 files that contain calo data.
    """

    def __init__(
        self,
        data_path: str,
        max_tracks: Optional[int] = None,
        pz_min: Optional[float] = None,
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

        self.track_index: List[Tuple[int, int]] = []
        n_filtered = 0
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, "r") as f:
                n = int(f.attrs.get("n_tracks", len(f["offsets"]) - 1))
                # Load pz column once for the whole file if filtering is needed
                if pz_min is not None and "mom_xyz" in f:
                    pz_col = f["mom_xyz"][:n, 2]  # shape (n,), pz is index 2
                else:
                    pz_col = None
                for ti in range(n):
                    if pz_col is not None and pz_col[ti] <= pz_min:
                        n_filtered += 1
                        continue
                    self.track_index.append((file_idx, ti))
                    if max_tracks and len(self.track_index) >= max_tracks:
                        break
            if max_tracks and len(self.track_index) >= max_tracks:
                break
        if n_filtered:
            logging.getLogger(__name__).info(
                f"FlatMomentumDataset: filtered out {n_filtered} backward-going tracks (pz <= {pz_min}); "
                f"{len(self.track_index)} forward-going tracks retained."
            )

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int):
        file_idx, track_idx = self.track_index[idx]
        f = _get_h5_handle(self.h5_files[file_idx])
        a = int(f["offsets"][track_idx])
        b = int(f["offsets"][track_idx + 1])
        features   = np.asarray(f["features"][a:b],    dtype=np.float32)
        ch_indices = np.asarray(f["ch_indices"][a:b],  dtype=np.int64)
        mom        = np.asarray(f["mom_xyz"][track_idx], dtype=np.float32)

        if not self.use_calo:
            return features, ch_indices, torch.from_numpy(mom)

        # --- Tracker + calorimeter mode ---
        # calo_scalars: (n_calo_feats,) float32, NaN where unmatched
        # calo_matched: scalar bool
        raw_calo = np.asarray(f["calo_scalars"][track_idx], dtype=np.float32)
        matched  = bool(f["calo_matched"][track_idx])

        # Replace NaN with 0 so the tensor is finite; append matched flag (0/1)
        # so the model can distinguish "zero because unmatched" from "zero because
        # the cluster genuinely had that value".
        raw_calo = np.nan_to_num(raw_calo, nan=0.0)
        calo_vec = np.append(raw_calo, float(matched)).astype(np.float32)

        return features, ch_indices, torch.from_numpy(mom), torch.from_numpy(calo_vec)


class MultiFileMomentumDataset(Dataset):
    """PyTorch Dataset for 3D momentum prediction from multiple HDF5 files."""

    def __init__(
        self,
        data_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t_diff", "edep"),
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
    feature_names: Tuple[str, ...] = ("t_diff", "edep"),
    task: str = "reconstruction",
    max_tracks: Optional[int] = None,
    output_dir: Optional[str] = None,
    flat_format: bool = False,
    pz_min: Optional[float] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    if flat_format:
        # Flat CSR format: __getitem__ is two array slices, no string decoding.
        if task == "momentum":
            use_calo = kwargs.get("use_calo", False)
            train_dataset = FlatMomentumDataset(train_data_path, max_tracks, pz_min=pz_min, use_calo=use_calo)
            val_dataset   = FlatMomentumDataset(val_data_path,   max_tracks, pz_min=pz_min, use_calo=use_calo)
        else:
            train_dataset = FlatHDF5Dataset(train_data_path, max_tracks)
            val_dataset   = FlatHDF5Dataset(val_data_path,   max_tracks)
    else:
        train_is_dir = Path(train_data_path).is_dir()
        val_is_dir   = Path(val_data_path).is_dir()

        # Cache the expensive HDF5 key scan so subsequent runs start instantly.
        train_cache = str(Path(output_dir) / "train_index.pkl") if output_dir else None
        val_cache   = str(Path(output_dir) / "val_index.pkl")   if output_dir else None

        if task == "momentum":
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    loss_fn: callable = masked_mse_loss,
    accumulation_steps: int = 1,
    task: str = "reconstruction",
    steps_per_epoch: Optional[int] = None,
) -> float:
    """Train for one epoch with AMP (automatic mixed precision)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = device.type == "cuda"
    amp_device = device.type  # "cuda" or "cpu"

    pbar = tqdm(dataloader, desc="Training", total=steps_per_epoch)
    for batch_idx, batch_data in enumerate(pbar):
        if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
            break
        if task == "momentum":
            # batch_data is either (x, mask, ch_idx, target) or
            # (x, mask, ch_idx, target, calo_scalars) depending on use_calo.
            has_calo_batch = len(batch_data) == 5
            if has_calo_batch:
                x, mask, channel_indices, target, calo_scalars = batch_data
                calo_scalars = calo_scalars.to(device, non_blocking=True)
            else:
                x, mask, channel_indices, target = batch_data
                calo_scalars = None
            x, mask, channel_indices, target = (
                x.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                channel_indices.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            with autocast(amp_device, enabled=use_amp):
                output = model(x, mask=mask, channel_indices=channel_indices,
                               calo_scalars=calo_scalars)
                loss = loss_fn(output, target) / accumulation_steps
        else:
            x, mask, channel_indices = batch_data
            x, mask, channel_indices = (
                x.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                channel_indices.to(device, non_blocking=True),
            )
            with autocast(amp_device, enabled=use_amp):
                output = model(x, mask=mask, channel_indices=channel_indices)
                loss = loss_fn(output, x, mask) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return total_loss / num_batches


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

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_data in pbar:
            if task == "momentum":
                has_calo_batch = len(batch_data) == 5
                if has_calo_batch:
                    x, mask, channel_indices, target, calo_scalars = batch_data
                    calo_scalars = calo_scalars.to(device, non_blocking=True)
                else:
                    x, mask, channel_indices, target = batch_data
                    calo_scalars = None
                x, mask, channel_indices, target = (
                    x.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                    channel_indices.to(device, non_blocking=True),
                    target.to(device, non_blocking=True),
                )
                with autocast(amp_device, enabled=use_amp):
                    output = model(x, mask=mask, channel_indices=channel_indices,
                                   calo_scalars=calo_scalars)
                    loss = loss_fn(output, target)
            else:
                x, mask, channel_indices = batch_data
                x, mask, channel_indices = (
                    x.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                    channel_indices.to(device, non_blocking=True),
                )
                with autocast(amp_device, enabled=use_amp):
                    output = model(x, mask=mask, channel_indices=channel_indices)
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
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-channels", type=int, default=41472)
    parser.add_argument("--model-type", choices=["reconstruction", "denoising", "momentum"], default="reconstruction")
    parser.add_argument("--loss-fn", choices=["mse", "l1"], default="mse")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4, help="Boost this to utilize multi-core CPU loading")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--infer-channel-index", type=int, default=None)
    parser.add_argument("--pooling-type", choices=["mean", "max", "attention"], default="mean")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Enable torch.compile (dynamic=True). Adds ~30s startup but can speed up large-GPU runs.")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Cap training batches per epoch. Useful when dataset is very large (e.g. --steps-per-epoch 2000).")
    parser.add_argument("--flat-format", action="store_true", default=False,
                        help="Use flat CSR HDF5 format produced by preprocess_to_flat.py (faster loading).")
    parser.add_argument("--forward-only", action="store_true", default=False,
                        help="Restrict training and validation to forward-going tracks (pz > 0). "
                             "Eliminates the sign ambiguity in pz that the model cannot resolve "
                             "from hit patterns alone. Only applies with --flat-format.")
    parser.add_argument("--use-calo", action="store_true", default=False,
                        help="Augment the momentum prediction head with calorimeter cluster scalars. "
                             "Requires flat HDF5 files produced from data with calo extraction enabled. "
                             "Only applies with --flat-format --model-type momentum. "
                             "When disabled (default), training is tracker-only.")

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs")

    logger.info(f"Device: {device}")
    
    train_data_resolved = _resolve_data_paths(args.train_data)
    val_data_resolved = _resolve_data_paths(args.val_data)

    logger.info("Building channel index...")
    if args.infer_channel_index:
        channel_index = infer_channel_index_from_hdf5(train_data_resolved, max_tracks=args.infer_channel_index)
    else:
        channel_index = {f"{p}_{pa}_{l}_{s}": idx for idx, (p, pa, l, s) in enumerate([(p, pa, l, s) for p in range(36) for pa in range(6) for l in range(2) for s in range(96)])}

    pz_min = 0.0 if args.forward_only else None
    if args.forward_only:
        logger.info("--forward-only enabled: restricting dataset to tracks with pz > 0.")

    use_calo = args.use_calo and args.flat_format and args.model_type == "momentum"
    if args.use_calo and not use_calo:
        logger.warning("--use-calo requires --flat-format and --model-type momentum; ignoring.")
    if use_calo:
        logger.info("--use-calo enabled: calorimeter scalars will be concatenated after pooling.")

    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data_resolved, val_data_resolved, channel_index,
        batch_size=args.batch_size, num_workers=args.num_workers, task=args.model_type,
        max_tracks=args.max_tracks, output_dir=str(output_dir), flat_format=args.flat_format,
        pz_min=pz_min, use_calo=use_calo,
    )
    logger.info(f"Train tracks: {len(train_loader.dataset):,} | Val tracks: {len(val_loader.dataset):,}")
    logger.info(f"Train batches/epoch: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # Determine calo_dim: number of calo scalar features + 1 for the calo_matched flag.
    # When use_calo=False, calo_dim=0 and the model is identical to the tracker-only version.
    if use_calo:
        # Peek at the first flat file to read n_calo_scalars from its metadata.
        first_file = train_loader.dataset.h5_files[0]
        with h5py.File(first_file, "r") as _f:
            n_calo_scalars = int(_f.attrs.get("n_calo_scalars", 0))
        calo_dim = n_calo_scalars + 1  # +1 for the appended calo_matched flag
        logger.info(f"calo_dim = {calo_dim} ({n_calo_scalars} calo scalars + 1 matched flag)")
    else:
        calo_dim = 0

    config = TrackModelConfig(
        input_dim=2, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, max_channels=args.max_channels,
        task=args.model_type, pooling_type=args.pooling_type, calo_dim=calo_dim,
    )

    if args.model_type == "denoising":
        model = DenoisingTrackModel(**config.to_dict())
    else:
        model = TrackReconstructionModel(**config.to_dict())

    # cuDNN auto-tuner: finds the fastest conv algorithm for fixed input sizes.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)

    # torch.compile can speed up training on large GPUs, but causes repeated
    # recompilation when sequence lengths vary batch-to-batch (dynamic shapes).
    # Only enable it with --compile; use torch.compile(..., dynamic=True) to
    # avoid per-shape recompilation at the cost of some peak performance.
    if args.compile:
        logger.info("Applying torch.compile(dynamic=True) to model...")
        model = torch.compile(model, dynamic=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    # Warmup scheduler steps per-batch; cosine scheduler steps per-epoch.
    warmup_steps = args.warmup_epochs * len(train_loader)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, args.epochs - args.warmup_epochs), T_mult=1, eta_min=1e-6)

    best_val_loss = float("inf")
    patience, patience_counter = 15, 0
    history = {"train_loss": [], "val_loss": []}

    loss_fn = momentum_loss if args.model_type == "momentum" else (masked_mse_loss if args.loss_fn == "mse" else masked_l1_loss)

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            loss_fn=loss_fn, accumulation_steps=args.accumulation_steps, task=args.model_type,
            steps_per_epoch=args.steps_per_epoch,
        )
        val_loss = validate(model, val_loader, device, loss_fn=loss_fn, task=args.model_type)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Step warmup per-batch during warmup epochs; then cosine per-epoch.
        if epoch < args.warmup_epochs:
            for _ in range(len(train_loader)):
                warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()