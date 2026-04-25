"""
Danyal Ahmed - April 2026

extra_analysis.py
Supplementary analysis for the domain-shift report. Runs AFTER evaluate_domain.py
and uses the same data paths. Produces three things that weren't in the main eval:

    1. precision_recall_by_redshift_bin.csv
       Per-class precision, recall, F1 per redshift bin per (model, domain).
       Answers: does per-class "accuracy" (= recall) and precision diverge on OOD?
       Specifically for class 3 (face-on non-spiral), recall rises with z while
       precision collapses — that's the over-prediction artifact.

    2. degradation_slopes.csv
       Linear fit of macro-F1 vs z_center on OOD bins 2-7 per model.
       Answers: which model degrades fastest per unit redshift?

    3. bin1_controlled_comparison.csv
       Bin-1 (z ≈ 0.089) IID-vs-OOD head-to-head.
       Answers: at identical redshift, how much does pure domain membership
       (easy-vs-hard classification) cost each model?

Usage:
    python extra_analysis.py \
        --svm-run output/runs/domain_svm \
        --gbdt-run output/runs/domain_gbdt \
        --cnn-run output/runs/domain_ce_lr3e-4_bs128_e30 \
        --output-dir output/domain_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from constants import CLASS_NAMES, N_CLASSES, OOD_SPLIT_COL
from dataset import GZ2Dataset, build_eval_transform
from evaluate import load_model_from_run
from evaluate_domain import (
    DomainPayload, RedshiftBins, build_redshift_bins,
    load_classical_backend, MIN_BIN_SIZE,
)


#################### HELPERS ####################

def load_cnn_payloads(
    run_dir: Path, iid_csv: Path, ood_csv: Path, image_dir: Path,
    stats_path: Path, batch_size: int = 128, num_workers: int = 4
) -> tuple[DomainPayload, DomainPayload]:
    """Run CNN inference. Mirror of evaluate_domain.load_cnn_backend but
    kept local so this script can be self-contained."""
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    model, _ = load_model_from_run(run_dir, checkpoint="best", device=device)

    stats = json.loads(stats_path.read_text())
    mean, std = stats["normalization"]["mean"], stats["normalization"]["std"]
    tf = build_eval_transform(mean, std)

    def _infer(csv_path: Path, domain: str) -> DomainPayload:
        ds = GZ2Dataset(csv_path=csv_path, image_dir=image_dir, transform=tf,
                        meta_cols=["dr7objid", OOD_SPLIT_COL])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device == "cuda"))
        all_preds, all_true, all_probs, all_oids, all_z = [], [], [], [], []
        print(f"    Running CNN inference on {csv_path.name} ({len(ds):,} images)...")
        with torch.no_grad():
            for imgs, hard, _soft, meta in loader:
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
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return iid, ood


#################### PRECISION/RECALL/F1 BY BIN ####################

def precision_recall_by_bin(
    payload: DomainPayload, bins: RedshiftBins
) -> pd.DataFrame:
    """Per-class precision, recall, F1 per redshift bin.

    Returns DataFrame columns:
        model, domain, bin, z_center, n, class, class_name,
        precision, recall, f1, support (= true count of this class in the bin)
    """
    bin_idx = bins.assign(payload.redshifts)
    rows = []
    for b in range(bins.n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n < MIN_BIN_SIZE:
            continue  # skip tiny bins entirely in this table for readability
        t = payload.true[mask]
        p = payload.preds[mask]
        pr, rc, f1, sup = precision_recall_fscore_support(
            t, p, labels=list(range(N_CLASSES)), zero_division=0
        )
        for k in range(N_CLASSES):
            rows.append({
                "model": payload.model, "domain": payload.domain,
                "bin": b, "z_center": float(bins.centers[b]), "n": n,
                "class": k, "class_name": CLASS_NAMES[k],
                "precision": float(pr[k]), "recall": float(rc[k]),
                "f1": float(f1[k]), "support": int(sup[k]),
            })
    return pd.DataFrame(rows)


#################### DEGRADATION SLOPES ####################

def compute_degradation_slopes(
    payloads: list[DomainPayload], bins: RedshiftBins,
    domain_filter: str = "ood"
) -> pd.DataFrame:
    """Linear fit of macro-F1 vs z_center for each model, over valid OOD bins, and
    fits per-class slopes, so the table shows where each model loses ground
    fastest.
    """
    rows = []
    for p in payloads:
        if p.domain != domain_filter:
            continue
        bin_idx = bins.assign(p.redshifts)

        # Collect per-bin macro-F1 and per-class recall 
        z_vals, f1_vals = [], []
        per_class_vals = {k: [] for k in range(N_CLASSES)}
        for b in range(bins.n_bins):
            mask = bin_idx == b
            n = int(mask.sum())
            if n < MIN_BIN_SIZE:
                continue
            t = p.true[mask]
            pr = p.preds[mask]
            from sklearn.metrics import f1_score
            z_vals.append(float(bins.centers[b]))
            f1_vals.append(float(f1_score(t, pr, average="macro",
                                           labels=list(range(N_CLASSES)),
                                           zero_division=0)))
            for k in range(N_CLASSES):
                cm = (t == k)
                if cm.sum() == 0:
                    per_class_vals[k].append(np.nan)
                else:
                    per_class_vals[k].append(float((pr[cm] == k).mean()))

        if len(z_vals) < 2:
            continue
        z_arr = np.array(z_vals)
        f1_arr = np.array(f1_vals)
        slope, intercept = np.polyfit(z_arr, f1_arr, 1)
        row = {
            "model": p.model, "domain": p.domain,
            "n_bins_used": len(z_vals),
            "macro_f1_slope": float(slope),
            "macro_f1_intercept": float(intercept),
        }
        for k in range(N_CLASSES):
            vals = np.array(per_class_vals[k], dtype=float)
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                row[f"class_{k}_slope"] = float("nan")
                continue
            s, _ = np.polyfit(z_arr[valid], vals[valid], 1)
            row[f"class_{k}_slope"] = float(s)
        rows.append(row)
    return pd.DataFrame(rows)


#################### BIN-1 CONTROLLED COMPARISON ####################

def bin1_controlled_comparison(
    payloads: list[DomainPayload], bins: RedshiftBins,
    target_bin: int = 1
) -> pd.DataFrame:
    """Head-to-head IID vs OOD at a single redshift bin."""
    rows = []
    for p in payloads:
        bin_idx = bins.assign(p.redshifts)
        mask = bin_idx == target_bin
        n = int(mask.sum())
        if n < MIN_BIN_SIZE:
            rows.append({
                "model": p.model, "domain": p.domain, "bin": target_bin,
                "n": n, "macro_f1": float("nan"), "accuracy": float("nan"),
                **{f"acc_class_{k}": float("nan") for k in range(N_CLASSES)}
            })
            continue
        t = p.true[mask]
        pr = p.preds[mask]
        from sklearn.metrics import f1_score
        per_class = {}
        for k in range(N_CLASSES):
            cm = (t == k)
            per_class[f"acc_class_{k}"] = (
                float("nan") if cm.sum() == 0 else float((pr[cm] == k).mean())
            )
        rows.append({
            "model": p.model, "domain": p.domain, "bin": target_bin,
            "n": n, "z_center": float(bins.centers[target_bin]),
            "macro_f1": float(f1_score(t, pr, average="macro",
                                        labels=list(range(N_CLASSES)),
                                        zero_division=0)),
            "accuracy": float((pr == t).mean()),
            **per_class,
        })
    return pd.DataFrame(rows)


#################### MAIN ####################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svm-run", type=Path, required=True)
    ap.add_argument("--gbdt-run", type=Path, required=True)
    ap.add_argument("--cnn-run", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, default=Path("data/gz2/processed/splits/domain"))
    ap.add_argument("--easyhard-dir", type=Path, default=Path("data/gz2/processed/splits/easyhard"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/gz2/images"))
    ap.add_argument("--output-dir", type=Path, default=Path("output/domain_eval"))
    ap.add_argument("--n-redshift-bins", type=int, default=8)
    ap.add_argument("--controlled-bin", type=int, default=1,
                    help="Bin where IID+OOD both have substantial samples. "
                         "With 8 quantile bins this is usually bin 1.")
    args = ap.parse_args()

    iid_csv = args.splits_dir / "test.csv"
    ood_csv = args.easyhard_dir / "hard.csv"
    stats_path = args.splits_dir / "stats.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load payloads
    print("Loading SVM predictions...")
    svm_iid, svm_ood = load_classical_backend("svm", args.svm_run, iid_csv, ood_csv)
    print("Loading GBDT predictions...")
    gbdt_iid, gbdt_ood = load_classical_backend("gbdt", args.gbdt_run, iid_csv, ood_csv)
    print("Running CNN inference...")
    cnn_iid, cnn_ood = load_cnn_payloads(
        args.cnn_run, iid_csv, ood_csv, args.image_dir, stats_path
    )
    payloads = [svm_iid, svm_ood, gbdt_iid, gbdt_ood, cnn_iid, cnn_ood]

    # Build same bins as main eval so results align
    pooled_z = np.concatenate([svm_iid.redshifts, svm_ood.redshifts])
    bins = build_redshift_bins(pooled_z, args.n_redshift_bins)
    print(f"\nRedshift bins (same as main eval): centers = "
          f"{[f'{c:.4f}' for c in bins.centers]}")

    # precision/recall/f1 per bin 
    print("\nComputing per-class precision/recall/F1 per bin...")
    pr_rows = [precision_recall_by_bin(p, bins) for p in payloads]
    pr_df = pd.concat(pr_rows, ignore_index=True)
    pr_df.to_csv(args.output_dir / "precision_recall_by_redshift_bin.csv", index=False)
    print(f"    Wrote {args.output_dir / 'precision_recall_by_redshift_bin.csv'}")

    # Analysis 2: degradation slopes 
    print("\nComputing degradation slopes (OOD, macro-F1 + per-class)...")
    slopes_df = compute_degradation_slopes(payloads, bins, domain_filter="ood")
    slopes_df.to_csv(args.output_dir / "degradation_slopes.csv", index=False)
    print(f"    Wrote {args.output_dir / 'degradation_slopes.csv'}")

    #  Analysis 3: bin-1 controlled comparison 
    print(f"\nBuilding bin-{args.controlled_bin} controlled IID-vs-OOD comparison...")
    bin1_df = bin1_controlled_comparison(payloads, bins, target_bin=args.controlled_bin)
    bin1_df.to_csv(args.output_dir / f"bin{args.controlled_bin}_controlled_comparison.csv",
                   index=False)
    print(f"    Wrote {args.output_dir / f'bin{args.controlled_bin}_controlled_comparison.csv'}")

    # summary
    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("Extra Analyses — Summary")
    summary_lines.append("=" * 78)

    # Slope summary
    summary_lines.append("")
    summary_lines.append("-" * 78)
    summary_lines.append("Macro-F1 degradation slope per unit redshift (OOD bins only)")
    summary_lines.append("-" * 78)
    summary_lines.append(f"{'model':<8}{'slope':>12}{'intercept':>12}  "
                         f"{'class_0_slope':>15}{'class_1_slope':>15}"
                         f"{'class_2_slope':>15}{'class_3_slope':>15}")
    for _, r in slopes_df.iterrows():
        summary_lines.append(
            f"{r['model']:<8}{r['macro_f1_slope']:>+12.3f}{r['macro_f1_intercept']:>+12.3f}  "
            f"{r['class_0_slope']:>+15.3f}{r['class_1_slope']:>+15.3f}"
            f"{r['class_2_slope']:>+15.3f}{r['class_3_slope']:>+15.3f}"
        )

    # Controlled comparison summary
    summary_lines.append("")
    summary_lines.append("-" * 78)
    summary_lines.append(f"Bin {args.controlled_bin} controlled comparison "
                         f"(IID vs OOD at same redshift)")
    summary_lines.append("-" * 78)
    summary_lines.append(f"{'model':<8}{'domain':<8}{'n':>8}  "
                         f"{'macro_f1':>10}{'accuracy':>10}  "
                         f"{'ellip':>8}{'edge':>8}{'spiral':>8}{'nonspir':>8}")
    for _, r in bin1_df.iterrows():
        summary_lines.append(
            f"{r['model']:<8}{r['domain']:<8}{r['n']:>8}  "
            f"{r['macro_f1']:>10.4f}{r['accuracy']:>10.4f}  "
            f"{r['acc_class_0']:>8.3f}{r['acc_class_1']:>8.3f}"
            f"{r['acc_class_2']:>8.3f}{r['acc_class_3']:>8.3f}"
        )

    # Deltas row for controlled comparison
    summary_lines.append("")
    summary_lines.append("IID -> OOD delta at same redshift (controlled):")
    for model in ["svm", "gbdt", "cnn"]:
        iid_row = bin1_df[(bin1_df["model"] == model) & (bin1_df["domain"] == "iid")]
        ood_row = bin1_df[(bin1_df["model"] == model) & (bin1_df["domain"] == "ood")]
        if iid_row.empty or ood_row.empty:
            continue
        d_f1 = float(ood_row["macro_f1"].iloc[0] - iid_row["macro_f1"].iloc[0])
        d_acc = float(ood_row["accuracy"].iloc[0] - iid_row["accuracy"].iloc[0])
        summary_lines.append(f"    {model:<6}  delta_macro_f1={d_f1:+.4f}  delta_accuracy={d_acc:+.4f}")

    # Non-spiral precision vs recall highlight — the over-prediction artifact
    summary_lines.append("")
    summary_lines.append("-" * 78)
    summary_lines.append("Face-on non-spiral (class 3) precision vs recall across OOD bins")
    summary_lines.append("(Recall rising while precision falls = over-prediction artifact)")
    summary_lines.append("-" * 78)
    summary_lines.append(f"{'model':<8}{'z':>8}  "
                         f"{'precision':>11}{'recall':>11}{'f1':>11}{'support':>10}")
    ns = pr_df[(pr_df["domain"] == "ood") & (pr_df["class"] == 3)].sort_values(["model", "z_center"])
    for _, r in ns.iterrows():
        summary_lines.append(
            f"{r['model']:<8}{r['z_center']:>8.3f}  "
            f"{r['precision']:>11.3f}{r['recall']:>11.3f}{r['f1']:>11.3f}{r['support']:>10}"
        )

    summary_lines.append("")
    summary_lines.append("=" * 78)
    (args.output_dir / "extra_analyses_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"    Wrote {args.output_dir / 'extra_analyses_summary.txt'}")
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()