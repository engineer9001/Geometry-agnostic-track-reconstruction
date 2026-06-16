import argparse
from pathlib import Path
from typing import Tuple
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from model import TrackReconstructionModel, TrackModelConfig
from train import (
    MultiFileMomentumDataset,
    MomentumTrackDataset,
    FlatMomentumDataset,
    sparse_collate_fn,
)
from torch.utils.data import DataLoader


def load_history(output_dir: Path):
    """Loads the training/validation loss history."""
    history_path = output_dir / "history.json"
    if not history_path.exists():
        print(f"Warning: No history.json found in {output_dir}")
        return None
    with open(history_path, "r") as f:
        return json.load(f)


def generate_predictions(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference to gather true vs predicted 3D momentum vectors."""
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, mask, channel_indices, target = batch
            x, mask, channel_indices = x.to(device), mask.to(device), channel_indices.to(device)
            
            # Output shape: (batch_size, 3)
            output = model(x, mask=mask, channel_indices=channel_indices)
            
            all_preds.append(output.cpu().numpy())
            all_trues.append(target.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)


def make_plots(history, v_true, v_pred, save_dir: Path):
    """Generates comprehensive validation dashboards using crisp 1D and 2D histograms."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------ PLOT 1: LOSS HISTORY ------------------
    if history:
        plt.figure(figsize=(7, 5))
        plt.plot(history["train_loss"], label="Train Loss", color="dodgerblue", lw=2)
        plt.plot(history["val_loss"], label="Val Loss", color="crimson", lw=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.title("Training History", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / "loss_history.png", dpi=200)
        plt.close()

    # Calculate 3D component residuals
    residuals = v_pred - v_true  # (N, 3) -> [dx, dy, dz]
    
    # Calculate Total Momentum Magnitudes using L2-norm
    p_true_tot = np.linalg.norm(v_true, axis=1)
    p_pred_tot = np.linalg.norm(v_pred, axis=1)
    residual_tot = p_pred_tot - p_true_tot

    components = ["X", "Y", "Z"]
    colors = ["dodgerblue", "crimson", "forestgreen"]

    # ------------------ PLOT 2: 1D COMPONENT RESIDUALS ------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (comp, color) in enumerate(zip(components, colors)):
        mean_res = np.mean(residuals[:, i])
        std_res = np.std(residuals[:, i])
        
        axes[i].hist(residuals[:, i], bins=50, color=color, edgecolor="black", alpha=0.7)
        axes[i].axvline(0, color="black", linestyle="-", alpha=0.5)
        axes[i].set_xlabel(f"Residual $\\Delta P_{{{comp.lower()}}}$ = $P_{{{comp.lower()},pred}}$ - $P_{{{comp.lower()},true}}$ [MeV/c]", fontsize=11)
        axes[i].set_ylabel("Counts", fontsize=11)
        axes[i].set_title(f"$P_{comp.lower()}$ 1D Residual\nMean={mean_res:.3f}, Std={std_res:.3f}", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_dir / "component_1d_residuals.png", dpi=200)
    plt.close()

    # ------------------ PLOT 3: 2D COMPONENT CORRELATIONS ------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, comp in enumerate(components):
        # Using cmin=1 leaves empty bins perfectly white
        counts, xedges, yedges, im = axes[i].hist2d(
            v_true[:, i], v_pred[:, i], bins=50, cmap="viridis", cmin=1, norm=LogNorm()
        )
        # Add ideal 45-degree reference line
        mn, mx = min(v_true[:, i]), max(v_true[:, i])
        axes[i].plot([mn, mx], [mn, mx], color="black", linestyle="--", alpha=0.7, label="Ideal")
        
        axes[i].set_xlabel(f"True $P_{comp.lower()}$ [MeV/c]", fontsize=11)
        axes[i].set_ylabel(f"Predicted $P_{comp.lower()}$ [MeV/c]", fontsize=11)
        axes[i].set_title(f"$P_{comp.lower()}$ 2D Distribution Histogram", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.2)
        fig.colorbar(im, ax=axes[i], label="Tracks per Bin")
        
    plt.tight_layout()
    plt.savefig(save_dir / "component_2d_correlations.png", dpi=200)
    plt.close()

    # ------------------ PLOT 4: TOTAL MOMENTUM DIAGNOSTICS ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Left: Total Momentum 2D Distribution Histogram
    counts, xedges, yedges, im = ax1.hist2d(
        p_true_tot, p_pred_tot, bins=50, cmap="plasma", cmin=1, norm=LogNorm()
    )
    mn, mx = min(p_true_tot), max(p_true_tot)
    ax1.plot([mn, mx], [mn, mx], color="black", linestyle="--", alpha=0.7, label="Ideal Perfect Fit")
    ax1.set_xlabel("True Total Momentum $P$ [MeV/c]", fontsize=11)
    ax1.set_ylabel("Predicted Total Momentum $P$ [MeV/c]", fontsize=11)
    ax1.set_title("Total Momentum 2D Correlation Histogram", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    fig.colorbar(im, ax=ax1, label="Tracks per Bin")
    ax1.legend(loc="upper left")

    # Right: Total Momentum 1D Residual Histogram
    mean_tot = np.mean(residual_tot)
    std_tot = np.std(residual_tot)
    ax2.hist(residual_tot, bins=50, color="purple", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="black", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Residual ($\Delta P = P_{pred} - P_{true}$) [MeV/c]", fontsize=11)
    ax2.set_ylabel("Counts", fontsize=11)
    ax2.set_title(f"Total Momentum 1D Residuals\nMean={mean_tot:.3f}, Std={std_tot:.3f}", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "total_momentum_diagnostics.png", dpi=200)
    plt.close()
    
    # ------------------ PLOT 5: TRUE vs PREDICTED 1D DISTRIBUTIONS WITH RATIO ------------------
    # Four panels: Px, Py, Pz, |P|_total
    quantities = [
        (v_true[:, 0], v_pred[:, 0], "$P_x$ [MeV/c]",     "dodgerblue"),
        (v_true[:, 1], v_pred[:, 1], "$P_y$ [MeV/c]",     "crimson"),
        (v_true[:, 2], v_pred[:, 2], "$P_z$ [MeV/c]",     "forestgreen"),
        (p_true_tot,   p_pred_tot,   "$|P|$ [MeV/c]",     "darkorange"),
    ]

    fig, axes_grid = plt.subplots(
        2, 4, figsize=(24, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex="col",
    )

    for col, (true_vals, pred_vals, xlabel, color) in enumerate(quantities):
        ax_main  = axes_grid[0, col]
        ax_ratio = axes_grid[1, col]

        # Shared bin edges covering both distributions
        lo = min(true_vals.min(), pred_vals.min())
        hi = max(true_vals.max(), pred_vals.max())
        bins = np.linspace(lo, hi, 60)
        centers = 0.5 * (bins[:-1] + bins[1:])

        n_true, _ = np.histogram(true_vals, bins=bins)
        n_pred, _ = np.histogram(pred_vals, bins=bins)

        # Poisson statistical errors: sqrt(N), with floor of 1 to avoid /0
        err_true = np.sqrt(np.maximum(n_true, 1))
        err_pred = np.sqrt(np.maximum(n_pred, 1))

        # Main panel: step histograms + error bars
        ax_main.step(bins, np.append(n_true, n_true[-1]), where="post",
                     color="black", lw=1.5, label="True")
        ax_main.errorbar(centers, n_true, yerr=err_true,
                         fmt="none", color="black", capsize=2, lw=1)

        ax_main.step(bins, np.append(n_pred, n_pred[-1]), where="post",
                     color=color, lw=1.5, linestyle="--", label="Predicted", alpha=0.85)
        ax_main.errorbar(centers, n_pred, yerr=err_pred,
                         fmt="none", color=color, capsize=2, lw=1, alpha=0.85)

        ax_main.set_ylabel("Tracks / Bin", fontsize=11)
        ax_main.set_title(xlabel, fontsize=13, fontweight="bold")
        ax_main.legend(fontsize=10)
        ax_main.grid(True, linestyle="--", alpha=0.4)
        ax_main.set_xlim(lo, hi)

        # Ratio panel: pred / true, with propagated uncertainty
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(n_true > 0, n_pred / n_true, np.nan)
            # Propagated relative error: sqrt((err_pred/n_pred)^2 + (err_true/n_true)^2) * ratio
            rel_err = np.where(
                (n_true > 0) & (n_pred > 0),
                ratio * np.sqrt((err_pred / np.maximum(n_pred, 1))**2 +
                                (err_true / np.maximum(n_true, 1))**2),
                np.nan,
            )

        ax_ratio.axhline(1.0, color="black", lw=1.2, linestyle="-")
        ax_ratio.fill_between(centers, 1 - 0.1, 1 + 0.1,
                              color="gray", alpha=0.15, label="±10%")
        ax_ratio.errorbar(centers, ratio, yerr=rel_err,
                          fmt="o", color=color, markersize=3, capsize=2, lw=1)

        ax_ratio.set_xlabel(xlabel, fontsize=11)
        ax_ratio.set_ylabel("Pred / True", fontsize=10)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, linestyle="--", alpha=0.4)
        ax_ratio.set_xlim(lo, hi)

    fig.suptitle("Predicted vs True Momentum Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_dir / "distribution_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Success! High-statistics tracking diagnostic dashboards saved to: {save_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D vector track tracking models.")
    parser.add_argument("test_data", help="Path to evaluation HDF5 folder or validation file split")
    parser.add_argument("--run-dir", required=True, help="Path containing best_model.pt and history.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--flat-format", action="store_true", default=False,
                        help="Use flat CSR HDF5 format produced by preprocess_to_flat.py.")
    parser.add_argument("--forward-only", action="store_true", default=False,
                        help="Restrict evaluation to forward-going tracks (pz > 0). "
                             "Should match the --forward-only flag used during training.")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    device = torch.device(args.device)

    # 1. Load weights & configuration metadata
    checkpoint = torch.load(run_path / "best_model.pt", map_location=device)
    config_dict = checkpoint["config"]
    config = TrackModelConfig(**config_dict)
    
    model = TrackReconstructionModel(**config.to_dict())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 2. Build standard tracker channel layout
    print("Building channel index mapping...")
    channel_index = {f"{p}_{pa}_{l}_{s}": idx for idx, (p, pa, l, s) in enumerate(
        [(p, pa, l, s) for p in range(36) for pa in range(6) for l in range(2) for s in range(96)]
    )}

    # 3. Create evaluation data loader
    pz_min = 0.0 if args.forward_only else None
    test_path = Path(args.test_data)
    if args.flat_format:
        test_dataset = FlatMomentumDataset(args.test_data, pz_min=pz_min)
    elif test_path.is_dir():
        test_dataset = MultiFileMomentumDataset(args.test_data, channel_index)
    else:
        test_dataset = MomentumTrackDataset(args.test_data, channel_index)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=sparse_collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    # 4. Extract arrays and generate plots
    print("Running evaluation network forward pass across test split entries...")
    history = load_history(run_path)
    v_pred, v_true = generate_predictions(model, test_loader, device)
    
    make_plots(history, v_true, v_pred, run_path / "plots")


if __name__ == "__main__":
    main()