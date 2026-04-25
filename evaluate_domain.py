"""
Danyal Ahmed - April 2026

evaluate_domain.py
Cross-model domain-shift evaluation

Loads trained instances of {svm, gbdt, cnn} and evaluates on both the IID test
set and OOD hard set. Produces:
    - metrics_comparison.csv
    - confusion_<model>_<domain>.png
    - perf_vs_redshift.png
    - per_class_vs_redshift.png
    - class_distribution_vs_redshift.png
    - summary.txt
    
Usage:
    [All Models]
    python evaluate_domain.py --models svm gbdt cnn \
        --svm-run output/runs/domain_svm \
        --gbdt-run output/runs/domain_gbdt \
        --cnn-run output/runs/domain_ce... \
        --output-dir output/domain_eval
        
    [Only Classical Models]
    python evaluate_domain.py --models svm gbdt \
        --svm-run output/runs/domain_svm \
        --gbdt-run output/runs/domain_gbdt \
        --output-dir output/domain_eval
        
PHASES (mostly for me writing this):
    - Load each model (classical features just load .npz, CNN runs inference)
    - Compute overall metrics and summary
    - Plot confusion matrices
    - Bin redshifts and plot redshift distributions against F1 overall, 
    F1 per-class, and per-class distribution
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from constants import CLASS_NAMES, N_CLASSES, CLASS_COLORS, OOD_SPLIT_COL
from dataset import GZ2Dataset, build_eval_transform
from evaluate import load_model_from_run


#################### CONSTANTS ####################

####                Model settings and info             ####
MODEL_CHOICES = ["svm", "gbdt", "cnn"]
MODEL_DISPLAY = {
    "svm": "Linear SVM",
    "gbdt": "Gradient Boosted Trees",
    "cnn": "Basic CNN"
}
MODEL_PLOT_COLORS = {   # from coolors.io
        "svm": "#1f77b4",
    "gbdt": "#2ca02c",
    "cnn": "#d62728"
}

####                Defaults                ####
DEFAULT_FEATURES_DIR = Path("data/gz2/processed/splits/domain")
DEFAULT_SPLITS_DIR = Path("data/gz2/processed/splits/domain")
DEFAULT_EASYHARD_DIR = Path("data/gz2/processed/splits/easyhard")
DEFAULT_IMAGE_DIR = Path("data/gz2/images")
DEFAULT_OUTPUT_DIR = Path("output/domain_eval")
DEFAULT_REDSHIFT_BINS = 8
DEFAULT_BOOTSTRAP_ITERS = 200

# Minimum per-bin sample count to avoid plotting poorly-populated values
MIN_BIN_SIZE = 30


#################### MODEL LOADING ####################

@dataclass
class DomainPayload:
    """Per-model, per-domain prediction payload.
    
    All arrays aligned by row, probs is optional (np.ndarray | None)
    """
    model: str
    domain: str
    preds: np.ndarray
    true: np.ndarray
    probs: np.ndarray | None
    objids: np.ndarray
    redshifts: np.ndarray
 
    @property
    def n(self) -> int:
        """Returns N
        
        Did this because it was getting annoying to type len(payload.preds) every time
        """
        return len(self.preds)
    

def _join_redshift(objids: np.ndarray, csv_path: Path) -> np.ndarray:
    """Joins redshift values from a source CSV onto an array of object ids

    Args:
        objids (np.ndarray): (n,) dr7objid's
        csv_path (Path): path to CSV with 'dr7objid' and OOD_SPLIT_COL (REDSHIFT)

    Returns:
        np.ndarray: (n,) float32 redshift value in same order as objids. NaN for
        any objid not found in the CSV
    """
    df = pd.read_csv(csv_path, usecols=["dr7objid", OOD_SPLIT_COL])
    
    # I know this is slow, but given the limited number of object ids and small 
    # size of the op, it shouldn't be too bad. May change to merge() later
    lookup = dict(zip(df["dr7objid"].to_numpy(dtype=np.int64), df[OOD_SPLIT_COL].to_numpy(dtype=np.float32)))
    
    # z is the general symbol for redshift
    z = np.array([lookup.get(int(oid), np.nan) for oid in objids], dtype=np.float32)
    
    n_missing = np.isnan(z).sum()
    if n_missing:
        print(f"    WARNING: {n_missing} objids not found in {csv_path.name} (redshift will be NaN for these rows)")
        
    return z


def load_classical_backend(
    model_name: str, run_dir: Path, iid_csv: Path, ood_csv: Path
) -> tuple[DomainPayload, DomainPayload]:
    """Load predictions_test.npz and predictions_ood.npz for a classical model

    Args:
        model_name (str): 'svm' or 'gbdt'
        run_dir (Path): directory with predictions_test.npz and predictions_ood.npz
        iid_csv (Path): in-domain testing csv
        ood_csv (Path): out-of-domain testing csv

    Returns:
        tuple[DomainPayload, DomainPayload]: (iid_payload, ood_payload)
    """
    iid_path = run_dir / "predictions_test.npz"
    ood_path = run_dir / "predictions_ood.npz"
    
    for p in (iid_path, ood_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing predictions: {p}")
        
    def _load(npz_path: Path, domain: str, source_csv: Path) -> DomainPayload:
        data = np.load(npz_path)
        probs = data["probs"] if "probs" in data.files else None
        objids = data["objids"].astype(np.int64)
        z = _join_redshift(objids, source_csv)
        
        return DomainPayload(
            model=model_name, domain=domain, preds=data['pred'].astype(np.int64),
            true=data["true"].astype(np.int64), probs=probs, objids=objids, redshifts=z
        )
        
    print(f"\nLoading {model_name.upper()} predictions from {run_dir}")
    iid = _load(iid_path, "iid", iid_csv)
    ood = _load(ood_path, "ood", ood_csv)
    print(f"    IID: n={iid.n:,}    probs={'yes' if iid.probs is not None else 'no'}")
    print(f"    OOD: n={ood.n:,}    probs={'yes' if ood.probs is not None else 'no'}")
    
    return iid, ood


def load_cnn_backend(
    run_dir: Path, iid_csv: Path, ood_csv: Path, image_dir: Path,
    stats_path: Path, batch_size: int, num_workers: int, checkpoint: str = "best"
) -> tuple[DomainPayload, DomainPayload]:
    """Run a trained CNN on IID and OOD test sets

    Args:
        run_dir (Path): CNN run directory (contains config.json and checkpoint_<checkpoint>.pt)
        iid_csv (Path): in-domain test csv
        ood_csv (Path): out-of-domain test csv
        image_dir (Path): directory with <asset_id>.jpg files
        stats_path (Path): stats.json for domain split (for normalization)
        batch_size (int): inference batch size
        num_workers (int): how many DataLoader workers
        checkpoint (str, optional): "best" or "last" model in training run. Defaults to "best".

    Returns:
        tuple[DomainPayload, DomainPayload]: (iid_payload, ood_payload)
    """
    print(f"\n Loading CNN from {run_dir}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg = load_model_from_run(run_dir, checkpoint=checkpoint, device=device)
    
    stats = json.loads(stats_path.read_text())
    mean = stats["normalization"]["mean"]
    std = stats["normalization"]["std"]
    tf = build_eval_transform(mean, std)
    
    def _infer(csv_path: Path, domain: str) -> DomainPayload:
        ds = GZ2Dataset(
            csv_path=csv_path, image_dir=image_dir, transform=tf, meta_cols=["dr7objid", OOD_SPLIT_COL]
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        all_preds, all_true, all_probs, all_oids, all_z = [], [], [] ,[], []
        print(f"    Running inference on {csv_path.name} ({len(ds):,} images)...")
        with torch.no_grad():
            for imgs, hard, _, meta in loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_preds.append(probs.argmax(axis=1))
                all_true.append(hard.numpy())
                all_oids.append(np.asarray(meta["dr7objid"], dtype=np.int64))
                all_z.append(np.asarray(meta[OOD_SPLIT_COL], dtype=np.float32))
                
        return DomainPayload(
            model="cnn", domain=domain,
            preds=np.concatenate(all_preds).astype(np.int64),
            true=np.concatenate(all_true).astype(np.int64),
            probs=np.concatenate(all_probs).astype(np.float32),
            objids=np.concatenate(all_oids).astype(np.int64),
            redshifts=np.concatenate(all_z).astype(np.float32),
        )
        
    iid = _infer(iid_csv, "iid")
    ood = _infer(ood_csv, "ood")
    print(f"    IID: n={iid.n:,}")
    print(f"    OOD: n={ood.n:,}")
    
    # Free GPU memory if necessary before plotting
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return iid, ood


#################### METRICS ####################

def _macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    """Macro-averaged F1
    
    Zero division is set to 0 so empty classes don't cause a crash
    """
    return float(f1_score(true, pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0.0))


def _accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    """Overall accuracy. NaN on empty array"""
    if len(true) == 0:
        return float("nan")
    return float((true == pred).mean())


def _per_class_accuracy(true: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    """Accuracy within each class. NaN for classes with no true samples"""
    out = {}
    for k in range(N_CLASSES):
        mask = (true == k)
        if mask.sum() == 0:
            out[k] == float("nan")
        else:
            out[k] = float((pred[mask] == k).mean())
            
    return out


def compute_payload_metrics(payload: DomainPayload) -> dict:
    """Gets overall and per-class metrics for one payload 

    Args:
        payload (DomainPayload): payload for one model and domain

    Returns:
        dict: (n, macro_f1, accuracy, per_class_accuracy (dict), per_class_n (dict), confusion_matrix (2D list) )
    """
    matrix = confusion_matrix(payload.true, payload.preds, labels=list(range(N_CLASSES)))
    per_class_n = {k: int((payload.true == k).sum()) for k in range(N_CLASSES)}
    
    return {
        "n" : int(payload.n),
        "macro_f1": _macro_f1(payload.true, payload.preds),
        "accuracy": _accuracy(payload.true, payload.preds),
        "per_class_accuracy": _per_class_accuracy(payload.true, payload.preds),
        "per_class_n": per_class_n,
        "confusion_matrix": matrix.tolist()
    }
    
    
#################### REDSHIFT BINNING ####################

@dataclass
class RedshiftBins:
    """
    Quantile-based redshift bins. Shared across models and domains for
    reasonable comparison.
    """
    
    edges: np.ndarray
    centers: np.ndarray
    n_bins: int
    
    def assign(self, z: np.ndarray) -> np.ndarray:
        """
        Return the bin index in [0, n_bins] for each redshift in z. 
        NaN -> -1.
        """
        idx = np.digitize(z, self.edges[1:-1], right=False)
        idx = np.where(np.isnan(z), -1, idx).astype(np.int64)
        return idx
    
    
def build_redshift_bins(all_redshifts: np.ndarray, n_bins: int) -> RedshiftBins:
    """Build the quantile bins from pooled redshift distribution (iid U ood)

    Uses quantiles to keep per-bin sample counts balanced, to stabilize macro-F1

    Args:
        all_redshifts (np.ndarray): pooled redshift across all payloads
                                    of the reference model (should be same across
                                    all models using the same dataset)
        n_bins (int): desired bin count

    Returns:
        RedshiftBins
    """
    z = all_redshifts[~np.isnan(all_redshifts)] # Remove NaNs
    
    if len(z) < n_bins * 5:
        raise ValueError(f"Too few reshift values {len(z)} to build {n_bins} bins. Reduce --redshift-bins")
        
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    
    # Nudge duplicate edges in the case of ties so that they don't get collapsed
    # Not sure how likely this actually is but felt right
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i-1] + 1e-6
            
    centers = np.zeros(n_bins)
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        in_bin = z[(z >= low) & (z <= high)] if i == n_bins - 1 else z[(z>=low) & (z < high)]   # off-by-one errors are the bane of my existence
        centers[i] = np.median(in_bin) if len(in_bin) else (low + high) / 2

    return RedshiftBins(edges=edges, centers=centers, n_bins=n_bins)


def _bootstrap_macro_f1_ci(
    true: np.ndarray, pred: np.ndarray, n_iter: int, seed: int = 0
) -> tuple[float, float]:
    """Percentile-based bootstrap in the 95% confidence interval (2.5th percentile - 97.5th percentile)
    
    NaN if n < 30
    
    Returns:
        (low, high) confidence interval
    """
    n = len(true)
    if n < 30:
        return float("nan"), float("nan")
    
    rng = np.random.default_rng(seed)
    scores = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        scores[i] = _macro_f1(true[idx], pred[idx])
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def metrics_by_redshift_bin(
    payload: DomainPayload, bins: RedshiftBins, bootstrap_iter: int = 0
) -> pd.DataFrame:
    """Compute macro-F1 + per-class accuracy + counts per redshift bin.
    
    Args:
        payload (DomainPayload): model predictions with redshifts attached
        bins (RedshiftBins): pre-built shared bins
        bootstrap_iter (int): if greater than 0, ci_low and ci_high columns are 
                              added for F1 score
                              
    Returns:
        pd.DataFrame: (
            model, domain, bin, z_center, n, macro_f1, accuracy, acc_class_0,
            acc_class_1, acc_class_2, acc_class_3, [ci_low, ci_high if bootstrap_iter > 0]
        )
    """
    bin_idx = bins.assign(payload.redshifts)
    rows = []
    for b in range(bins.n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        
        row = {
            "model": payload.model,
            "domain": payload.domain,
            "bin": b,
            "z_center": float(bins.centers[b]),
            "n": n
        }
        
        # Honestly might just change this to drop these rows entirely
        if n < MIN_BIN_SIZE:
            row.update({
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                **{f"acc_class_{k}": float("nan") for k in range(N_CLASSES)}
            })
            if bootstrap_iter > 0:
                row["ci_low"] = float("nan")
                row["ci_high"] = float("nan")
        else:
            t = payload.true[mask]
            p = payload.preds[mask]
            pc = _per_class_accuracy(t, p)
            row.update({
                "macro_f1": _macro_f1(t, p),
                "accuracy": _accuracy(t, p),
                **{f"acc_class_{k}": pc[k] for k in range(N_CLASSES)}
            })
            
            if bootstrap_iter > 0:
                low, high = _bootstrap_macro_f1_ci(t, p, bootstrap_iter, seed=b)
                row["ci_low"] = low
                row["ci_high"] = high
                
        rows.append(row)
        
    return pd.DataFrame(rows)


#################### OUTPUT SUMMARIES ####################
 
def write_metrics_comparison_csv(
    payloads: list[DomainPayload], out_path: Path
) -> None:
    """Writes full metrics CSV covering overall + per-class for all (model, domain).
 
    Columns: model, domain, scope, class, n, accuracy, macro_f1
        scope='overall' -> class='all'
        scope='per_class' -> class=class name, macro_f1=NaN (per-class macro-F1 isn't meaningful)
    """
    rows = []
    for p in payloads:
        m = compute_payload_metrics(p)
        rows.append({
            "model": p.model, "domain": p.domain, "scope": "overall", "class": "all",
            "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
        })
        for k in range(N_CLASSES):
            rows.append({
                "model": p.model, "domain": p.domain, "scope": "per_class",
                "class": CLASS_NAMES[k],
                "n": m["per_class_n"][k],
                "accuracy": m["per_class_accuracy"][k],
                "macro_f1": float("nan"),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    Wrote {out_path}")
 
 
def write_summary_text(
    payloads: list[DomainPayload], out_path: Path
) -> None:
    """Human-readable summary.
    
    Probably the most tedious thing I've written so far and it isn't even
    that crucial
    """
    # Group payloads by model for easy lookup
    by_model = {}
    for p in payloads:
        by_model.setdefault(p.model, {})[p.domain] = p
 
    models = [m for m in MODEL_CHOICES if m in by_model]
 
    lines = []
    lines.append("=" * 78)
    lines.append("Domain-Shift Evaluation Summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Models evaluated: {', '.join(MODEL_DISPLAY[m] for m in models)}")
    for m in models:
        iid, ood = by_model[m].get("iid"), by_model[m].get("ood")
        if iid is not None and ood is not None:
            lines.append(f"    {MODEL_DISPLAY[m]:<24}  IID n={iid.n:>6,}  OOD n={ood.n:>6,}")
 
    # Overall macro-F1
    lines.append("")
    lines.append("-" * 78)
    lines.append("Overall macro-F1")
    lines.append("-" * 78)
    lines.append(f"{'Model':<26}  {'IID':>8}  {'OOD':>8}  {'degrade':>10}")
    for m in models:
        iid = by_model[m].get("iid")
        ood = by_model[m].get("ood")
        if iid is None or ood is None:
            continue
        f_iid = _macro_f1(iid.true, iid.preds)
        f_ood = _macro_f1(ood.true, ood.preds)
        lines.append(f"{MODEL_DISPLAY[m]:<26}  {f_iid:>8.4f}  {f_ood:>8.4f}  {f_iid - f_ood:>+10.4f}")
 
    # Overall accuracy
    lines.append("")
    lines.append("-" * 78)
    lines.append("Overall accuracy")
    lines.append("-" * 78)
    lines.append(f"{'Model':<26}  {'IID':>8}  {'OOD':>8}  {'degrade':>10}")
    for m in models:
        iid = by_model[m].get("iid")
        ood = by_model[m].get("ood")
        if iid is None or ood is None:
            continue
        a_iid = _accuracy(iid.true, iid.preds)
        a_ood = _accuracy(ood.true, ood.preds)
        lines.append(f"{MODEL_DISPLAY[m]:<26}  {a_iid:>8.4f}  {a_ood:>8.4f}  {a_iid - a_ood:>+10.4f}")
 
    # Per-class accuracy, IID then OOD
    for domain_label in ("iid", "ood"):
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"Per-class accuracy ({domain_label.upper()})")
        lines.append("-" * 78)
        header = f"{'Class':<22}  " + "  ".join(f"{MODEL_DISPLAY[m][:12]:>12}" for m in models)
        lines.append(header)
        for k in range(N_CLASSES):
            vals = []
            for m in models:
                p = by_model[m].get(domain_label)
                if p is None:
                    vals.append("       -    ")
                    continue
                pc = _per_class_accuracy(p.true, p.preds)
                vals.append(f"{pc[k]:>12.4f}")
            lines.append(f"{CLASS_NAMES[k]:<22}  " + "  ".join(vals))
 
    # Per-class degradation
    lines.append("")
    lines.append("-" * 78)
    lines.append("Per-class accuracy change (IID -> OOD, positive = improvement)")
    lines.append("-" * 78)
    header = f"{'Class':<22}  " + "  ".join(f"{MODEL_DISPLAY[m][:12]:>12}" for m in models)
    lines.append(header)
    for k in range(N_CLASSES):
        vals = []
        for m in models:
            iid, ood = by_model[m].get("iid"), by_model[m].get("ood")
            if iid is None or ood is None:
                vals.append("       -    ")
                continue
            a_iid = _per_class_accuracy(iid.true, iid.preds)[k]
            a_ood = _per_class_accuracy(ood.true, ood.preds)[k]
            vals.append(f"{a_ood - a_iid:>+12.4f}")
        lines.append(f"{CLASS_NAMES[k]:<22}  " + "  ".join(vals))
 
    lines.append("")
    lines.append("=" * 78)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    Wrote {out_path}")
    
    
#################### OUTPUT PLOTS ####################

def plot_confusion_matrix(payload: DomainPayload, out_path: Path) -> None:
    """Creates a confusion matrix with both raw counts and row-normalized values"""
    m = compute_payload_metrics(payload)
    matrix = np.array(m["confusion_matrix"])
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / np.maximum(row_sums, 1)
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)
    
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text = f"{matrix[i, j]}\n({matrix_norm[i, j]:.2f})"
            color = "white" if matrix_norm[i,j] > 0.5 else "black"      # makes sure text contrasts the background
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
            
            
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([CLASS_NAMES[k] for k in range(N_CLASSES)], rotation=30, ha="right")
    ax.set_yticklabels([CLASS_NAMES[k] for k in range(N_CLASSES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{MODEL_DISPLAY[payload.model]} : {payload.domain.upper()}\n"
        f"macro-F1={m['macro_f1']:.4f}  accuracy={m['accuracy']:.4f}  n={m['n']:,}"
    )
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    
def plot_performance_vs_redshift(
    per_bin_df: pd.DataFrame, z_cut: float, out_path: Path
) -> None:
    """Macro-F1 vs redshift bin centers. 
    
    Bootstrap CI shaded, original data cut highlighted
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    has_ci = "ci_low" in per_bin_df.columns
    
    # Since I don't have raw prediction probabilities, IID and OOD are plotted
    # as separate series per-model
    for model in per_bin_df["model"].unique():
        for domain in ("iid", "ood"):
            # Subset with this model and domain
            sub = per_bin_df[(per_bin_df["model"] == model) & 
                             (per_bin_df["domain"] == domain)].sort_values("z_center")
            
            if len(sub) == 0:
                continue
            
            valid = ~sub["macro_f1"].isna()
            if valid.sum() == 0:
                continue
            
            # in domain solid, out of domain dashed
            ls = "-" if domain == "iid" else "--"
            label = f"{MODEL_DISPLAY[model]} ({domain.upper()})"
            color = MODEL_PLOT_COLORS[model]
            ax.plot(sub.loc[valid, "z_center"], sub.loc[valid, "macro_f1"],
                    marker="o", ls=ls, color=color, label=label, alpha=0.9)
            if has_ci:
                ax.fill_between(
                    sub.loc[valid, "z_center"],
                    sub.loc[valid, "ci_low"], sub.loc[valid, "ci_high"],
                    color=color, alpha=0.12
                )
                
    ax.axvline(z_cut, color="black", ls=":", lw=1.2, alpha=0.7, label=f"easy/hard cutoff z={z_cut:.3f}")
    ax.set_xlabel("Redshift (bin median)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Performance vs Redshift")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    

def plot_acc_per_class_vs_redshift(
    per_bin_df: pd.DataFrame, z_cut: float, out_path: Path
) -> None:
    """2x2 grid of per-class accuracy vs redshift bin, one subplot per-class"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for k in range(N_CLASSES):
        ax = axes[k]
        col = f"acc_class_{k}"
        
        for model in per_bin_df["model"].unique():
            for domain in ("iid", "ood"):
                sub = per_bin_df[(per_bin_df["model"] == model) &
                                 (per_bin_df["domain"] == domain)].sort_values("z_center")
                
                if len(sub) == 0:
                    continue
                
                valid = ~sub[col].isna()
                if valid.sum() == 0:
                    continue
                
                ls = "-" if domain == "iid" else "--"
                ax.plot(sub.loc[valid, "z_center"], sub.loc[valid, col],
                        marker="o", ls=ls, color=MODEL_PLOT_COLORS[model],
                        label=f"{MODEL_DISPLAY[model]} ({domain.upper()})", alpha=0.9)
                
        ax.axvline(z_cut, color="black", ls=":", lw=1, alpha=0.6)
        ax.set_title(CLASS_NAMES[k])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if k >= 2:
            ax.set_xlabel("Redshift (bin median)")
        if k % 2 == 0:
            ax.set_ylabel("Per-class accuracy")
            
    # legend outside the grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=min(len(labels), 4), fontsize=8)
    fig.suptitle("Per-Class Accuracy vs. Redshift", y=1.06, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_class_distribution_vs_redshift(
    reference_payloads: list[DomainPayload], bins: RedshiftBins,
    z_cut: float, out_path: Path
) -> None:
    """Stacked area showing class mix by redshift bin
    
    Not really a model plot, but shows data property so I can see if performance
    may be due to class distribution over actual design choices
    """
    # pool labels
    z_all = np.concatenate([p.redshifts for p in reference_payloads])
    y_all = np.concatenate([p.true for p in reference_payloads])
    bin_idx = bins.assign(z_all)
    
    # fraction of each class in each bin
    fracs = np.zeros((bins.n_bins, N_CLASSES))
    totals = np.zeros(bins.n_bins, dtype=int)
    for b in range(bins.n_bins):
        mask = bin_idx == b
        
        totals[b] = int(mask.sum())
        if totals[b] == 0:
            continue
        
        for k in range(N_CLASSES):
            fracs[b, k] = ((y_all[mask] == k).sum()) / totals[b]
            
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    bottom = np.zeros(bins.n_bins)
    
    for k in range(N_CLASSES):
        ax1.fill_between(bins.centers, bottom, bottom + fracs[:, k],
                         color=CLASS_COLORS[k], label=CLASS_NAMES[k], alpha=0.85)
        
        bottom += fracs[:, k]
        
    ax1.axvline(z_cut, color="black", ls=":", lw=1.2, alpha=0.7, label=f"easy/hard cutoff z={z_cut:.3f}")
    ax1.set_ylabel("Class fraction within redshift bin")
    ax1.set_ylim(0, 1)
    ax1.set_title("Class Composition vs. Redshift (pooled IID + OOD)")
    ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def _read_z_cut(meta_path: Path) -> float:
    """Read ood_split_z_cut from split_meta.txt, or fall back to NaN."""
    if not meta_path.exists():
        print(f"    WARNING: {meta_path} not found, z-cut annotation will be NaN")
        return float("nan")
    for line in meta_path.read_text().splitlines():
        if line.startswith("ood_split_z_cut"):
            return float(line.split("=")[1].strip())
    return float("nan")
    
    
#################### MAIN ####################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, required=True,
                        help="Which models to evaluate (any subset)")
    parser.add_argument("--svm-run", type=Path, default=None,
                        help="Path to SVM run directory (required if --models svm)")
    parser.add_argument("--gbdt-run", type=Path, default=None,
                        help="Path to GBDT run directory (required if --models gbdt)")
    parser.add_argument("--cnn-run", type=Path, default=None,
                        help="Path to CNN run directory (required if --models cnn)")
    
    # Directory options
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                        help=f"Directory with test.csv + stats.json (default: {DEFAULT_SPLITS_DIR})")
    parser.add_argument("--easyhard-dir", type=Path, default=DEFAULT_EASYHARD_DIR,
                        help=f"Directory with hard.csv (default: {DEFAULT_EASYHARD_DIR})")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help=f"Image directory, used by CNN backend (default: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
 
    parser.add_argument("--redshift-bins", type=int, default=DEFAULT_REDSHIFT_BINS,
                        help=f"Number of redshift bins for the vs-redshift plots (default: {DEFAULT_REDSHIFT_BINS})")
    parser.add_argument("--bootstrap-iter", type=int, default=DEFAULT_BOOTSTRAP_ITERS,
                        help=f"Bootstrap iterations for macro-F1 CI (default: {DEFAULT_BOOTSTRAP_ITERS}, 0 to skip)")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best",
                        help="CNN checkpoint to load (default: best)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="CNN inference batch size (default: 128)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="CNN DataLoader workers (default: 4)")
 
    args = parser.parse_args()
    
    # Check paths
    required_runs = {"svm": args.svm_run, "gbdt": args.gbdt_run, "cnn": args.cnn_run}
    for m in args.models:
        if required_runs[m] is None:
            print(f"ERROR: --{m}-run is required when {m} is in --models")
            sys.exit(1)
        if not required_runs[m].exists():
            print(f"ERROR: --{m}-run path does not exist: {required_runs[m]}")
            sys.exit(1)
            
    args.output_dir.mkdir(parents=True, exist_ok=True)
    iid_csv = args.splits_dir / "test.csv"
    ood_csv = args.easyhard_dir / "hard.csv"
    stats_path = args.splits_dir / "stats.json"
    
    for required, label in [(iid_csv, "IID test CSV"), (ood_csv, "OOD hard CSV"), (stats_path, "stats.json")]:
        if not required.exists():
            print(f"ERROR: {label} not found at {required}")
            sys.exit(1)
            
    # get z cutoff, fallback to default from constants.py if not there
    z_cut = _read_z_cut(args.easyhard_dir / "split_meta.txt")
    print(f"easy/hard cutoff: z = {z_cut}")
    
    # Load models
    payloads: list[DomainPayload] = []
    for m in args.models:
        if m in ("svm", "gbdt"):
            iid, ood = load_classical_backend(m, required_runs[m], iid_csv, ood_csv)
        elif m == "cnn":
            iid, ood, = load_cnn_backend(
                required_runs[m], iid_csv, ood_csv, args.image_dir, stats_path,
                batch_size=args.batch_size, num_workers=args.num_workers, checkpoint=args.checkpoint
            )
        else:
            continue
        
        payloads.extend([iid, ood])
        
        
    # Metrics and summary
    print(f"\nWriting tables to {args.output_dir}")
    write_metrics_comparison_csv(payloads, args.output_dir / "metrics_comparison.csv")
    write_summary_text(payloads, args.output_dir / "summary.txt")
    
    # Confusion matriecs
    print(f"\nGenerating confusion matrices...")
    for p in payloads:
        out = args.output_dir / f"confusion_{p.model}_{p.domain}.png"
        plot_confusion_matrix(p, out)
        print(f"    Wrote {out}")
        
    # redshift bin analysis
    print("\nBuilding redshift bins...")
    # Since all models see the same galaxies, just use the first model's payloads as reference
    ref_iid = next(p for p in payloads if p.domain == "iid")
    ref_ood = next(p for p in payloads if p.domain == "ood")
    pooled_z = np.concatenate([ref_iid.redshifts, ref_ood.redshifts])
    bins = build_redshift_bins(pooled_z, args.redshift_bins)
    print(f"    {bins.n_bins} quantile bins, centers = {bins.centers.round(4).tolist()}")
    
    per_bin_rows = []
    for p in payloads:
        per_bin_rows.append(
            metrics_by_redshift_bin(p, bins, bootstrap_iter=args.bootstrap_iter)
        )
    per_bin_df = pd.concat(per_bin_rows, ignore_index=True)
    per_bin_df.to_csv(args.output_dir / "metrics_by_redshift_bin.csv", index=False)
    print(f"    Wrote {args.output_dir / 'metrics_by_redshift_bin.csv'}")
    
    print("\nGenerating redshift plots...")
    plot_performance_vs_redshift(per_bin_df, z_cut, args.output_dir / "perf_vs_redshift.png")
    print(f"    Wrote {args.output_dir / 'perf_vs_redshift.png'}")
    plot_acc_per_class_vs_redshift(per_bin_df, z_cut, args.output_dir / "per_class_vs_redshift.png")
    print(f"    Wrote {args.output_dir / 'per_class_vs_redshift.png'}")
    plot_class_distribution_vs_redshift(
        [ref_iid, ref_ood], bins, z_cut,
        args.output_dir / "class_distribution_vs_redshift.png"
    )
    print(f"    Wrote {args.output_dir / 'class_distribution_vs_redshift.png'}")
 
    # print summary
    print("\n" + (args.output_dir / "summary.txt").read_text())
    
    
if __name__ == "__main__":
    main()