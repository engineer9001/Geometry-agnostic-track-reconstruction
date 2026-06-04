import argparse
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import TrackReconstructionModel, TrackModelConfig
from train import create_dataloaders


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
        axes[i].set_xlabel(f"Residual ($\Delta P_{comp.lower()} = P_{comp.lower(), pred} - P_{comp.lower(), true}$) [MeV/c]", fontsize=11)
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
            v_true[:, i], v_pred[:, i], bins=50, cmap="viridis", cmin=1
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
        p_true_tot, p_pred_tot, bins=50, cmap="plasma", cmin=1
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
    
    print(f"Success! High-statistics tracking diagnostic dashboards saved to: {save_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D vector track tracking models.")
    parser.add_argument("test_data", help="Path to evaluation HDF5 folder or validation file split")
    parser.add_argument("--run-dir", required=True, help="Path containing best_model.pt and history.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    # 2. Build standard tracker channel layout layout
    print("Building channel index mapping...")
    channel_index = {f"{p}_{pa}_{l}_{s}": idx for idx, (p, pa, l, s) in enumerate(
        [(p, pa, l, s) for p in range(36) for pa in range(6) for l in range(2) for s in range(96)]
    )}

    # 3. Create evaluation data loader
    _, test_loader = create_dataloaders(
        args.test_data, args.test_data, channel_index,
        batch_size=64, num_workers=2, task="momentum"
    )

    # 4. Extract arrays and generate plots
    print("Running evaluation network forward pass across test split entries...")
    history = load_history(run_path)
    v_pred, v_true = generate_predictions(model, test_loader, device)
    
    make_plots(history, v_true, v_pred, run_path / "plots")


if __name__ == "__main__":
    main()