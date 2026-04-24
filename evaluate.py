"""
Danyal Ahmed - April 2026

evaluate.py
For the analysis of trained models to compute primary metrics:
    - Top-1 accuracy, per-class accuracy, confusion matrix
    - Expected Calibration Error with reliability diagram
    - KL divergence between predictions and vote fractions
    
Metrics should also be stratified by label ambiguity (hard vs. soft)

Outputs in --output-dir:
    metrics.csv
    per_class_metrics.csv
    reliability.png
    confusion_<name>.png
    ece_by_stratum.png
"""
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from constants import CLASS_NAMES, N_CLASSES
from dataset import GZ2Dataset, build_eval_transform
from train import get_device
from models import BasicCNN


#################### CONSTANTS ####################

# Default number of bins for calibration analysis
DEFAULT_CAL_BINS = 10


#################### MODEL STRUCTURES ####################

@dataclass
class RunResults:
    """All results computed for a trained model on the test set
    
    Attributes:
        name (str): run directory name, for identifier
        loss_type (str): "ce" | "kl"
        probs (np.ndarray): softmax probabilities, (N, num_classes)
        preds (np.ndarray): argmax predictions, (N,)
        hard (np.ndarray): hard labels, (N,)
        soft (np.ndarray): vote-fraction soft labels, (N, num_classes)
    """
    name: str
    loss_type: str
    probs: np.ndarray
    preds: np.ndarray
    hard: np.ndarray
    soft: np.ndarray
    
    
def load_model_from_run(run_dir: Path, checkpoint: str = "best", device: str=get_device()) -> tuple[BasicCNN, dict]:
    """Load a trained BasicCNN from a run directory

    Args:
        run_dir (Path): directory containing config.json and checkpoint_[checkpoint].pt
        checkpoint (str, optional): which checkpoint to load ('best' or 'last'). Defaults to "best".
        device (str, optional): device for model. Defaults to get_device().

    Returns:
        tuple[BasicCNN, dict]: (model, config)
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    
    model = BasicCNN(num_classes=N_CLASSES, dropout=cfg.get('dropout', 0.4))
    checkpoint_path = run_dir / f"checkpoint_{checkpoint}.pt"
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    model.to(device)
    
    return model, cfg


@torch.no_grad()
def run_inference(model: BasicCNN, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on every batch in loader and collect probabilities, hard labels, and soft labels

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (probs, hard, soft)
            probs: (N, num_classes) float32 softmax probabilities
            hard: (N,) int64 hard labels
            soft: (N, num_classes) float32 vote-fraction soft labels
    """
    all_probs, all_hard, all_soft = [], [], []
    
    for imgs, hard, soft, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_hard.append(hard.numpy())
        all_soft.append(soft.numpy())
        
    return (
        np.concatenate(all_probs).astype(np.float32),
        np.concatenate(all_hard).astype(np.int64),
        np.concatenate(all_soft).astype(np.float32)
    )
    
    
#################### METRIC COMPUTATIONS ####################

def accuracy(preds: np.ndarray, hard: np.ndarray) -> float:
    """Top-1 accuracy"""
    return float((preds == hard).mean())


def per_class_accuracy(preds: np.ndarray, hard: np.ndarray) -> dict[int, float]:
    """Per-class accuracy as a dict of class_index -> accuracy"""
    out = {}
    for k in range(N_CLASSES):
        mask = (hard == k)
        if mask.sum() == 0:
            out[k] = 0.0
        else:
            out[k] = float((preds[mask] == k).mean())
            
    return out


def confusion_matrix(preds: np.ndarray, hard: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix as square ndarray where matrix[i,j]
    is count of examples with true class i predicted as class j
    """
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            matrix[i,j] = ((hard == i) & (preds == j)).sum()
            
    return matrix


def expected_calibration_error(
    probs: np.ndarray, hard: np.ndarray,
    n_bins: int = DEFAULT_CAL_BINS
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected calibration error with equal-width confidence bins

    Partitions predictions by their max-class confidence into bins from 0-1,
    then computes |avg_confidence - avg_accuracy| per bin

    Returns:
        tuple[float, np.ndarray, np.ndarray, np.ndarray]: (ece, bin_confidences, bin_accuracies, bin_counts)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == hard).astype(np.float64)
    
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n_bins):
        low, high = bin_bounds[i], bin_bounds[i + 1]
        # Last bin closes on the right to include 1.0
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
            
        count = mask.sum()
        bin_counts[i] = count
        
        if count > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = correct[mask].mean()
            
    n = len(confidences)
    ece = float(np.sum(
        bin_counts * np.abs(bin_confidences - bin_accuracies) / n
    ))
    
    return ece, bin_confidences, bin_accuracies, bin_counts


def brier_score(probs: np.ndarray, hard: np.ndarray) -> float:
    """Multi-class Brier score

    MSE between predicted prob and one-hot label summed across classes. Lower = better
    """
    one_hot = np.eye(N_CLASSES)[hard]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def kl_to_soft_targets(probs: np.ndarray, soft: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL divergence from soft targets to model predictions
    
    KL(q || p) = sum_k(q[k] * log(q[k] / p[k])) - q is soft target and p is predicted distribution.
    eps is a min value to avoid log(0)
    
    Essentially how well the prediction matches the human disagreement.
    """
    p = np.clip(probs, eps, 1.0)
    q = np.clip(soft, eps, 1.0)
    
    per_sample = (q * (np.log(q) - np.log(p))).sum(axis=1)
    return float(per_sample.mean())


def ambiguity_mask(soft: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Split examples into 'clean' or 'ambiguous' (max soft target >= cutoff)
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (clean_mask, ambiguous_mask)
    """
    max_confidence = soft.max(axis=1)
    clean_mask = max_confidence >= cutoff
    ambiguous_mask = ~clean_mask
    return clean_mask, ambiguous_mask


def per_class_stratified_ece(result: RunResults, ambiguity_cutoff: float
                              ) -> dict[tuple[int, str], tuple[float, int]]:
    """Compute ECE per (class, group) for one run.
    
    Returns:
        Dict mapping (class_index, group_name) -> (ece, n)
    """
    clean_mask, ambig_mask = ambiguity_mask(result.soft, ambiguity_cutoff)
    group = {
        "overall": np.ones(len(result.hard), dtype=bool),
        "clean":   clean_mask,
        "ambiguous": ambig_mask,
    }
    
    out = {}
    for k in range(N_CLASSES):
        class_mask = result.hard == k
        for group_name, group_mask in group.items():
            mask = class_mask & group_mask
            n = int(mask.sum())
            if n < 20:   # ECE unstable on tiny samples
                out[(k, group_name)] = (float("nan"), n)
                continue
            ece, _, _, _ = expected_calibration_error(
                result.probs[mask], result.hard[mask]
            )
            out[(k, group_name)] = (ece, n)
    return out


def per_class_stratified_accuracy(result, ambiguity_cutoff):
    clean_mask, ambig_mask = ambiguity_mask(result.soft, ambiguity_cutoff)
    strata = {"overall": np.ones(len(result.hard), dtype=bool),
              "clean": clean_mask, "ambiguous": ambig_mask}
    out = {}
    for k in range(N_CLASSES):
        class_mask = result.hard == k
        for group_name, group_mask in strata.items():
            mask = class_mask & group_mask
            n = int(mask.sum())
            if n == 0:
                out[(k, group_name)] = (float("nan"), 0)
                continue
            acc = float((result.preds[mask] == result.hard[mask]).mean())
            out[(k, group_name)] = (acc, n)
    return out

####                Metric Orchestration                ####

def compute_all_metrics(result: RunResults, ambiguity_cutoff: float) -> dict[str, dict[str, float]]:
    """Compute all metrics both overall and stratified by ambiguity

    Returns:
        dict[str, dict[str, float]]: 
            {amb_name: {metric_name: value}}
            
            amb_name = "overall" | "clean" | "ambiguous"
    """
    probs = result.probs
    preds = result.preds
    hard = result.hard
    soft = result.soft
    
    clean_mask, ambiguous_mask = ambiguity_mask(soft, ambiguity_cutoff)
    
    ambs = {
        "overall" : np.ones(len(hard), dtype=bool),
        "clean": clean_mask,
        "ambiguous": ambiguous_mask
    }
    
    out = {}
    for amb_name, mask in ambs.items():
        if mask.sum() == 0:
            continue
        
        p = probs[mask]
        pr = preds[mask]
        h = hard[mask]
        s = soft[mask]
        
        ece, _, _, _ = expected_calibration_error(p, h)
        
        metrics = {
            "n": int(mask.sum()),
            "accuracy": accuracy(pr, h),
            "ece": ece,
            "brier": brier_score(p, h),
            "kl_to_soft": kl_to_soft_targets(p, s)
        }
        out[amb_name] = metrics
        
    return out


#################### PLOTS ####################

def plot_reliability_diagrams(
    results: list[RunResults], output_path: Path, n_bins: int=DEFAULT_CAL_BINS
) -> None:
    """Side by side reliability diagrams for each run

    Plots model confidence against actual accuracy at that confidence
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes.flatten()
    
    for ax, result in zip(axes, results):
        ece, bin_conf, bin_acc, bin_counts = expected_calibration_error(
            result.probs, result.hard
        )
        
        nonempty = bin_counts > 0
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
        ax.bar(
            bin_conf[nonempty], bin_acc[nonempty],
            width=(1.0 / n_bins) * 0.9, align="center", color="steelblue", 
            alpha=0.7, edgecolor="black", linewidth=0.5, label="Observed"
        )
        
        # Show gap
        for conf, acc in zip(bin_conf[nonempty], bin_acc[nonempty]):
            ax.plot([conf, conf], [acc, conf], color="crimson", lw=0.8, alpha=0.6)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Accuracy in bin")
        ax.set_title(f"{result.name}\nECE = {ece:.4f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_confusion_matrix(result: RunResults, output_path: Path) -> None:
    """Confusion matrix as heatmap, with counts and row-normalizes percentages"""
    matrix = confusion_matrix(result.preds, result.hard)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / np.maximum(row_sums, 1)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)
    
    # Annotate each cell
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text = f"{matrix[i,j]}\n({matrix_norm[i,j]:.2f})"
            color = "white" if matrix_norm[i,j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
            
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([CLASS_NAMES[k] for k in range(N_CLASSES)], rotation=30, ha="right")
    ax.set_yticklabels([CLASS_NAMES[k] for k in range(N_CLASSES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {result.name}\n(counts and row-normalized fractions)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_ece_by_group(results: list[RunResults], all_metrics: dict, output_path: Path) -> None:
    """Bar chart comparing ECE across runs and groups.

    Groups: overall, clean, and ambiguous
    """
    groups = ["overall", "clean", "ambiguous"]
    run_names = [r.name for r in results]
    
    ece_values = np.zeros((len(groups), len(results)))
    for i, group in enumerate(groups):
        for j, result in enumerate(results):
            ece_values[i, j] = all_metrics[result.name][group]['ece']
            
    x = np.arange(len(groups))
    width = 0.8 / len(results)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, name in enumerate(run_names):
        offset = (j - (len(results) - 1) / 2) * width
        bars = ax.bar(x + offset, ece_values[:, j], width, label=name)
        for bar, v in zip(bars, ece_values[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in groups])
    ax.set_ylabel("ECE (lower=better)")
    ax.set_title("Expected Calibration Error by group")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_accuracy_by_group(results: list[RunResults], all_metrics: dict, output_path: Path) -> None:
    """Bar chart comparing top-1 accuracies across runs and group"""
    
    groups = ['overall', 'clean', "ambiguous"]
    run_names = [r.name for r in results]
    
    acc_values = np.zeros((len(groups), len(results)))
    for i, group in enumerate(groups):
        for j, result in enumerate(results):
            acc_values[i, j] = all_metrics[result.name][group]["accuracy"]
            
    x = np.arange(len(groups))
    width = 0.8 / len(results)
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, name in enumerate(run_names):
        offset = (j - (len(results) - 1) / 2) * width
        bars = ax.bar(x + offset, acc_values[:, j], width, label=name)
        for bar, v in zip(bars, acc_values[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in groups])
    ax.set_ylabel("Top-1 accuracy (high=better)")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by group")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
#################### CSV WRITERS ####################

def write_metrics_csv(all_metrics: dict, out_path: Path) -> None:
    """Write one row per (run, group) pair with all metrics"""
    rows = []
    for run_name, group_data in all_metrics.items():
        for group, metrics in group_data.items():
            row = {"run": run_name, "group": group, **metrics}
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    
def write_per_class_csv(results: list[RunResults], out_path: Path) -> None:
    """Overall per-class accuracy for each run"""
    rows = []
    for result in results:
        per_class = per_class_accuracy(result.preds, result.hard)
        row = {"run": result.name}
        for k in range(N_CLASSES):
            row[CLASS_NAMES[k]] = per_class[k]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    
def write_per_class_stratified_ece_csv(
    all_per_class_ece: dict[str, dict[tuple[int, str], tuple[float, int]]],
    out_path: Path,
) -> None:
    """Write per-class stratified ECE in long format."""
    rows = []
    for run_name, per_class_data in all_per_class_ece.items():
        for (k, group), (ece, n) in per_class_data.items():
            rows.append({
                "run":        run_name,
                "class":      k,
                "class_name": CLASS_NAMES[k],
                "group":    group,
                "ece":        ece,
                "n":          n,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    
def write_per_class_stratified_accuracy_csv(
    all_per_class_acc: dict[str, dict[tuple[int, str], tuple[float, int]]],
    out_path: Path,
) -> None:
    """Write per-class stratified accuracy in long format."""
    rows = []
    for run_name, per_class_data in all_per_class_acc.items():
        for (k, stratum), (acc, n) in per_class_data.items():
            rows.append({
                "run":        run_name,
                "class":      k,
                "class_name": CLASS_NAMES[k],
                "stratum":    stratum,
                "accuracy":   acc,
                "n":          n,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    
def write_reliability_bins_csv(results, out_path):
    rows = []
    for r in results:
        _, conf, acc, counts = expected_calibration_error(r.probs, r.hard)
        for bin_idx, (c, a, n) in enumerate(zip(conf, acc, counts)):
            rows.append({
                "run": r.name, "bin": bin_idx,
                "confidence": c, "accuracy": a, "count": n,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    
def write_summary_text(
    results: list[RunResults], all_metrics: dict, ambiguity_cutoff: float, out_path: Path
) -> None:
    """Human-readable summary of comparisons
    
    Added bc the CSVs were annoying to look at
    """
    lines = []
    lines.append("="*72)
    lines.append("Hard vs Soft Label Comparison")
    lines.append("=" * 72)
    lines.append(f"\nAmbiguity cutoff: max(soft_target) < {ambiguity_cutoff} = ambiguous")
    lines.append(f"Runs evaluated: {len(results)}")
    for r in results:
        lines.append(f"    {r.name:<40}  loss={r.loss_type}")
        
    lines.append("\n" + "-" * 72)
    lines.append("Overall test set")
    lines.append("-" * 72)
    lines.append(f"{'Metric':<20} " + " ".join(f"{r.name[:22]:>24}" for r in results))
    
    # Overall comparisons
    for metric in ["n", "accuracy", "ece", "brier", "kl_to_soft"]:
        vals = [all_metrics[r.name]["overall"][metric] for r in results]
        if metric == "n":
            val_strs = [f"{int(v):>24,}" for v in vals]
        else:
            val_strs = [f"{v:>24.4f}" for v in vals]
        lines.append(f"{metric:<20} " + " ".join(val_strs))
    
    # Group comparisons
    for group in ["clean", "ambiguous"]:
        lines.append("\n" + "-" * 72)
        lines.append(f"{group.capitalize()} group")
        lines.append("-" * 72)
        lines.append(f"{'Metric':<20} " + " ".join(f"{r.name[:22]:>24}" for r in results))
        for metric in ["n", "accuracy", "ece", "brier", "kl_to_soft"]:
            vals = [all_metrics[r.name][group][metric] for r in results]
            if metric == "n":
                val_strs = [f"{int(v):>24,}" for v in vals]
            else:
                val_strs = [f"{v:>24.4f}" for v in vals]
            lines.append(f"{metric:<20} " + " ".join(val_strs))
            
    # Per-class accuracy
    lines.append("\n" + "-" * 72)
    lines.append("Per-class accuracy (overall test set)")
    lines.append("-" * 72)
    lines.append(f"{'Class':<25} " + " ".join(f"{r.name[:22]:>24}" for r in results))
    for k in range(N_CLASSES):
        vals = [per_class_accuracy(r.preds, r.hard)[k] for r in results]
        val_strs = [f"{v:>24.4f}" for v in vals]
        lines.append(f"{CLASS_NAMES[k]:<25} " + " ".join(val_strs))
        
    lines.append("\n" + '=' * 72)
    
    out_path.write_text("\n".join(lines))
    
    
#################### MAIN ####################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run directories to compare")
    parser.add_argument("--splits-dir", required=True, help="Directory containing test.csv and stats.json")
    
    # Directory option
    parser.add_argument("--image-dir", default="data/gz2/images")
    parser.add_argument("--output-dir", default="output/comparisons", help="Where to write comparison artifacts")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best", help="Which checkpoint to evaluate from each run")
    parser.add_argument("--ambiguity-cutoff", type=float, default=0.8, help="max(soft_target) < cutoff counts as ambiguous")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=get_device())
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build shared test loader so runs evaluate on same test set
    splits_dir = Path(args.splits_dir)
    stats = json.loads((splits_dir / "stats.json").read_text())
    mean = stats["normalization"]["mean"]
    std = stats["normalization"]["std"]
    
    test_ds = GZ2Dataset(
        splits_dir / "test.csv", args.image_dir, transform=build_eval_transform(mean, std)
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0), pin_memory=True
    )
    
    print(f"Test set: {len(test_ds):,} examples")
    print(f"Output directory: {output_dir}\n")
    
    # Evaluate each run
    results: list[RunResults] = []
    for run_path_str in args.runs:
        run_dir = Path(run_path_str)
        run_name = run_dir.name
        
        print(f"Evaluating {run_name}...")
        model, cfg = load_model_from_run(run_dir, checkpoint=args.checkpoint, device=args.device)
        probs, hard, soft = run_inference(model, test_loader, args.device)
        preds = probs.argmax(axis=1)
        
        results.append(RunResults(
            name=run_name,
            loss_type=cfg.get("loss", "unknown"),
            probs=probs, preds=preds, hard=hard, soft=soft
        ))
        
        print(f"    accuracy = {accuracy(preds, hard):.4f}  ece = {expected_calibration_error(probs, hard)[0]:.4f}")
        
        # Free GPU memory before next run
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
            
    # Compute metrics
    print(f"\nComputing metrics...")
    all_metrics = {r.name: compute_all_metrics(r, args.ambiguity_cutoff) for r in results}
    
    all_per_class_ece = {r.name: per_class_stratified_ece(r, args.ambiguity_cutoff) for r in results}
    all_per_class_acc = {r.name: per_class_stratified_accuracy(r, args.ambiguity_cutoff) for r in results}
    
    # Write CSVs
    write_metrics_csv(all_metrics, output_dir / "metrics.csv")
    write_per_class_csv(results, output_dir / "per_class_metrics.csv")
    write_per_class_stratified_ece_csv(all_per_class_ece, output_dir / "per_class_stratified_ece.csv")
    write_per_class_stratified_accuracy_csv(all_per_class_acc, output_dir / "per_class_stratified_accuracy.csv")
    write_reliability_bins_csv(results, output_dir / "reliability.csv")
    
    # Summary text
    write_summary_text(results, all_metrics, args.ambiguity_cutoff, output_dir / "summary.txt")
    
    # Plots
    print("Generating plots...")
    plot_reliability_diagrams(results, output_dir / "reliability.png")
    plot_ece_by_group(results, all_metrics, output_dir / "ece_by_group.png")
    plot_accuracy_by_group(results, all_metrics, output_dir / "accuracy_by_group.png")
    
    for r in results:
        plot_confusion_matrix(r, output_dir / f"confusion_{r.name}.png")
        
    print(f"\n--- Comparison written to {output_dir} ---\n")
    print((output_dir / "summary.txt").read_text())
    

    
    
if __name__ == "__main__":
    main()