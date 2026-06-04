import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
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
    """
    has_momentum = len(batch[0]) == 3
    
    if has_momentum:
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

    if has_momentum:
        return x_padded, mask_padded, indices_padded, torch.stack(momentum_list)
    return x_padded, mask_padded, indices_padded


class MultiFileHDF5Dataset(Dataset):
    """PyTorch Dataset that loads sparse tracks from multiple HDF5 files."""

    def __init__(
        self,
        data_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t_diff", "edep"),
        max_tracks: Optional[int] = None,
    ):
        self.logger = logging.getLogger(__name__)
        p = Path(data_path)
        
        if p.is_dir():
            self.h5_files = sorted(p.glob("*.h5"))
        elif p.is_file():
            self.h5_files = [p]
        else:
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        self.channel_index = channel_index
        self.feature_names = feature_names

        self.track_index = []
        for file_idx, h5_file in enumerate(self.h5_files):
            try:
                with h5py.File(h5_file, "r") as f:
                    if "tracks" not in f:
                        continue
                    for track_name in f["tracks"].keys():
                        self.track_index.append((file_idx, track_name))
                        if max_tracks and len(self.track_index) >= max_tracks:
                            break
                    if max_tracks and len(self.track_index) >= max_tracks:
                        break
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file_idx, track_name = self.track_index[idx]
        with h5py.File(self.h5_files[file_idx], "r") as f:
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
        self.h5_path = h5_path
        self.channel_index = channel_index
        self.feature_names = feature_names

        with h5py.File(h5_path, "r") as f:
            self.track_names = list(f["tracks"].keys())
            if max_tracks:
                self.track_names = self.track_names[:max_tracks]

    def __len__(self) -> int:
        return len(self.track_names)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
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
        self.h5_path = h5_path
        self.channel_index = channel_index
        self.feature_names = feature_names

        with h5py.File(h5_path, "r") as f:
            self.track_names = list(f["tracks"].keys())
            if max_tracks:
                self.track_names = self.track_names[:max_tracks]

    def __len__(self) -> int:
        return len(self.track_names)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            track_group = f["tracks"][self.track_names[idx]]
            features, indices = build_sparse_track_from_hdf5_group(
                track_group, self.channel_index, self.feature_names
            )
            # Unpack individual vector components from group attributes
            px = track_group.attrs.get("true_mom_x", 0.0)
            py = track_group.attrs.get("true_mom_y", 0.0)
            pz = track_group.attrs.get("true_mom_z", 0.0)
            
        return features, indices, torch.tensor([px, py, pz], dtype=torch.float32)


class MultiFileMomentumDataset(Dataset):
    """PyTorch Dataset for 3D momentum prediction from multiple HDF5 files."""

    def __init__(
        self,
        data_path: str,
        channel_index: Dict[str, int],
        feature_names: Tuple[str, ...] = ("t_diff", "edep"),
        max_tracks: Optional[int] = None,
    ):
        p = Path(data_path)
        if p.is_dir():
            self.h5_files = sorted(p.glob("*.h5"))
        else:
            self.h5_files = [p]

        self.channel_index = channel_index
        self.feature_names = feature_names
        self.track_index = []

        for file_idx, h5_file in enumerate(self.h5_files):
            try:
                with h5py.File(h5_file, "r") as f:
                    if "tracks" not in f:
                        continue
                    for track_name in f["tracks"].keys():
                        self.track_index.append((file_idx, track_name))
                        if max_tracks and len(self.track_index) >= max_tracks:
                            break
                    if max_tracks and len(self.track_index) >= max_tracks:
                        break
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        file_idx, track_name = self.track_index[idx]
        with h5py.File(self.h5_files[file_idx], "r") as f:
            track_group = f["tracks"][track_name]
            features, indices = build_sparse_track_from_hdf5_group(
                track_group, self.channel_index, self.feature_names
            )
            # Unpack individual vector components from group attributes
            px = track_group.attrs.get("true_mom_x", 0.0)
            py = track_group.attrs.get("true_mom_y", 0.0)
            pz = track_group.attrs.get("true_mom_z", 0.0)
            
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
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_is_dir = Path(train_data_path).is_dir()
    val_is_dir = Path(val_data_path).is_dir()
    
    if task == "momentum":
        train_dataset = MultiFileMomentumDataset(train_data_path, channel_index, feature_names, max_tracks) if train_is_dir else MomentumTrackDataset(train_data_path, channel_index, feature_names, max_tracks)
        val_dataset = MultiFileMomentumDataset(val_data_path, channel_index, feature_names, max_tracks) if val_is_dir else MomentumTrackDataset(val_data_path, channel_index, feature_names, max_tracks)
    else:
        train_dataset = MultiFileHDF5Dataset(train_data_path, channel_index, feature_names, max_tracks) if train_is_dir else TrackHDF5Dataset(train_data_path, channel_index, feature_names, max_tracks)
        val_dataset = MultiFileHDF5Dataset(val_data_path, channel_index, feature_names, max_tracks) if val_is_dir else TrackHDF5Dataset(val_data_path, channel_index, feature_names, max_tracks)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=sparse_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=sparse_collate_fn,
    )
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: callable = masked_mse_loss,
    accumulation_steps: int = 1,
    task: str = "reconstruction",
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch_data in enumerate(pbar):
        if task == "momentum":
            x, mask, channel_indices, target = batch_data
            x, mask, channel_indices, target = x.to(device), mask.to(device), channel_indices.to(device), target.to(device)
            output = model(x, mask=mask, channel_indices=channel_indices)
            # Removed target.unsqueeze(1) to map perfectly with shape (batch_size, 3)
            loss = loss_fn(output, target)
        else:
            x, mask, channel_indices = batch_data
            x, mask, channel_indices = x.to(device), mask.to(device), channel_indices.to(device)
            output = model(x, mask=mask, channel_indices=channel_indices)
            loss = loss_fn(output, x, mask)

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

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

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_data in pbar:
            if task == "momentum":
                x, mask, channel_indices, target = batch_data
                x, mask, channel_indices, target = x.to(device), mask.to(device), channel_indices.to(device), target.to(device)
                output = model(x, mask=mask, channel_indices=channel_indices)
                # Removed target.unsqueeze(1) to map perfectly with shape (batch_size, 3)
                loss = loss_fn(output, target)
            else:
                x, mask, channel_indices = batch_data
                x, mask, channel_indices = x.to(device), mask.to(device), channel_indices.to(device)
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

    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data_resolved, val_data_resolved, channel_index,
        batch_size=args.batch_size, num_workers=args.num_workers, task=args.model_type, max_tracks=args.max_tracks,
    )

    config = TrackModelConfig(
        input_dim=2, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, max_channels=args.max_channels,
        task=args.model_type, pooling_type=args.pooling_type,
    )

    if args.model_type == "denoising":
        model = DenoisingTrackModel(**config.to_dict())
    else:
        model = TrackReconstructionModel(**config.to_dict())

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = args.warmup_epochs * len(train_loader)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=1, eta_min=1e-6)

    best_val_loss = float("inf")
    patience, patience_counter = 15, 0
    history = {"train_loss": [], "val_loss": []}

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn=momentum_loss if args.model_type == "momentum" else (masked_mse_loss if args.loss_fn == "mse" else masked_l1_loss), accumulation_steps=args.accumulation_steps, task=args.model_type)
        val_loss = validate(model, val_loader, device, loss_fn=momentum_loss if args.model_type == "momentum" else (masked_mse_loss if args.loss_fn == "mse" else masked_l1_loss), task=args.model_type)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch < args.warmup_epochs:
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