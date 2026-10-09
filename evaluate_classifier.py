import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TrackModelConfig, TrackReconstructionModel
from train import FlatClassifierDataset, sparse_collate_fn


def generate_predictions(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validation labels, raw logits, and sigmoid Ce scores."""
    labels, logits = [], []
    model.eval()
    with torch.no_grad():
        for x, mask, channels, target in tqdm(dataloader, desc="Validation inference"):
            output = model(
                x.to(device, non_blocking=True),
                mask=mask.to(device, non_blocking=True),
                channel_indices=channels.to(device, non_blocking=True),
            )
            labels.append(target.numpy().astype(np.int8))
            logits.append(output.float().cpu().numpy())
    if not labels:
        raise RuntimeError("Validation dataset is empty")
    label_array = np.concatenate(labels)
    logit_array = np.concatenate(logits)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logit_array, -80.0, 80.0)))
    return label_array, logit_array, scores


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute a tie-safe empirical ROC curve and trapezoidal AUC."""
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC requires both Ce and DIO validation samples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    tp = np.cumsum(sorted_labels)[distinct]
    fp = (distinct + 1) - tp
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    thresholds = np.r_[np.inf, sorted_scores[distinct]]
    auc = float(np.trapz(tpr, fpr) if hasattr(np, "trapz") else np.trapezoid(tpr, fpr))
    return fpr, tpr, thresholds, auc


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    pred = scores >= threshold
    true = labels.astype(bool)
    tn = int((~true & ~pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tp = int((true & pred).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def scalar_metrics(cm: np.ndarray, auc: float, threshold: float, n_tracks: int) -> Dict[str, object]:
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    safe = lambda num, den: float(num / den) if den else None
    return {
        "split": "validation",
        "positive_class": "CeEndpointOnSpill",
        "negative_class": "DIOtail95OnSpill",
        "threshold": threshold,
        "n_tracks": n_tracks,
        "n_ce": tp + fn,
        "n_dio": tn + fp,
        "roc_auc": auc,
        "accuracy": safe(tp + tn, n_tracks),
        "precision_ce": safe(tp, tp + fp),
        "recall_ce_signal_efficiency": safe(tp, tp + fn),
        "specificity_dio": safe(tn, tn + fp),
        "background_acceptance": safe(fp, tn + fp),
        "background_rejection": safe(tn + fp, fp) if fp else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def plot_confusion(cm: np.ndarray, output: Path, normalized: bool) -> None:
    shown = cm.astype(float)
    if normalized:
        shown /= np.maximum(shown.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    for row in range(2):
        for col in range(2):
            text = f"{shown[row, col]:.3f}" if normalized else f"{int(shown[row, col]):,}"
            ax.text(col, row, text, ha="center", va="center",
                    color="white" if shown[row, col] > shown.max() / 2 else "black")
    ax.set_xticks([0, 1], ["DIO", "Ce"])
    ax.set_yticks([0, 1], ["DIO", "Ce"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Validation confusion matrix" + (" (row normalized)" if normalized else ""))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)



def make_plots(
    labels: np.ndarray, scores: np.ndarray, fpr: np.ndarray, tpr: np.ndarray,
    auc: float, cm: np.ndarray, history: Dict[str, list], output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(scores[labels == 0], bins=bins, histtype="step", linewidth=2,
            label="DIO (negative)", color="tab:blue", density=True)
    ax.hist(scores[labels == 1], bins=bins, histtype="step", linewidth=2,
            label="Ce (positive)", color="tab:orange", density=True)
    ax.set(xlabel="Ce score", ylabel="Normalized tracks / bin",
           title="Ce-vs-DIO validation score distributions")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "score_histogram.png", dpi=200)
    ax.set_yscale("log")
    fig.savefig(output / "score_histogram_log.png", dpi=200)
    plt.close(fig)

    plot_confusion(cm, output / "confusion_matrix.png", normalized=False)
    plot_confusion(cm, output / "confusion_matrix_normalized.png", normalized=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"CVN classifier (AUC={auc:.5f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set(xlabel="DIO acceptance (false-positive rate)",
           ylabel="Ce efficiency (true-positive rate)",
           title="Ce-vs-DIO validation ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=200)
    plt.close(fig)

    rejection = np.divide(1.0, fpr, out=np.full_like(fpr, np.inf), where=fpr > 0)
    finite = np.isfinite(rejection)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tpr[finite], rejection[finite], linewidth=2)
    ax.set_yscale("log")
    ax.set(xlabel="Ce efficiency",
           ylabel="DIO rejection (1 / background acceptance)",
           title="Ce-vs-DIO validation background rejection")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(output / "background_rejection.png", dpi=200)
    plt.close(fig)

    if history and history.get("train_loss"):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, history["train_loss"], label="Training")
        axes[0].plot(epochs, history["val_loss"], label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Classifier loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        if history.get("train_accuracy"):
            axes[1].plot(epochs, history["train_accuracy"], label="Training")
            axes[1].plot(epochs, history["val_accuracy"], label="Validation")
            axes[1].legend()
        axes[1].set(xlabel="Epoch", ylabel="Accuracy at score 0.5",
                    title="Classifier accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "training_history.png", dpi=200)
        plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a cvn_classifier checkpoint on validation flat HDF5 data."
    )
    parser.add_argument("run_dir", help="Training run directory containing best_model.pt")
    parser.add_argument("validation_data", help="Flat validation HDF5 file or directory")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: RUN_DIR/best_model.pt)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: RUN_DIR/classifier_validation_plots)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--require-calo", action="store_true",
                        help="Evaluate only calo-matched tracks; does not use calo features.")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--no-predictions", action="store_true",
                        help="Do not save validation_predictions.npz.")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")

    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    output_dir = (Path(args.output_dir) if args.output_dir
                  else run_dir / "classifier_validation_plots")
    device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    if config_dict.get("task") != "cvn_classifier":
        raise ValueError(
            f"Expected cvn_classifier checkpoint, got task={config_dict.get('task')!r}"
        )
    model = TrackReconstructionModel(**TrackModelConfig(**config_dict).to_dict())
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.to(device)

    dataset = FlatClassifierDataset(
        args.validation_data, max_tracks=args.max_tracks,
        require_calo=args.require_calo,
    )
    if min(dataset.class_counts.values()) == 0:
        raise RuntimeError(f"Evaluation requires both classes; got {dataset.class_counts}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=sparse_collate_fn, persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    labels, logits, scores = generate_predictions(model, loader, device)
    fpr, tpr, thresholds, auc = roc_curve(labels, scores)
    cm = confusion(labels, scores, args.threshold)
    metrics = scalar_metrics(cm, auc, args.threshold, len(labels))
    metrics.update({
        "checkpoint": str(checkpoint_path.resolve()),
        "validation_data": str(Path(args.validation_data).resolve()),
        "require_calo": args.require_calo,
    })
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "roc_curve.npz", fpr=fpr, tpr=tpr, thresholds=thresholds
    )
    if not args.no_predictions:
        np.savez_compressed(
            output_dir / "validation_predictions.npz",
            labels=labels, logits=logits, ce_scores=scores,
        )
    make_plots(labels, scores, fpr, tpr, auc, cm, history, output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Validation metrics and plots saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
