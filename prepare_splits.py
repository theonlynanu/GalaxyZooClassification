"""
Danyal Ahmed - April 2026
 
prepare_splits.py
One-shot script to build stratified train/val/test splits for the CV final
project from the labeled GZ2 CSV produced by analysis.py.
 
1. Load gz2_labeled.csv (already filtered to confident, image-backed rows)
2. Stratified subsample to a target size (default 50,000)
3. Compute branch-product soft targets from the GZ2 decision tree and
    renormalize to sum to 1 across the four classes
4. Stratified 70/15/15 split into train/val/test with a fixed seed
5. Compute per-channel RGB mean/std from the TRAINING split images only
6. Write train.csv, val.csv, test.csv, and stats.json
 
Usage:
    python prepare_splits.py [labeled_csv] [image_dir] [out_dir] [--size N] [--seed S]
 
Default paths (match the current project layout):
    labeled_csv  = data/gz2/processed/gz2_labeled.csv
    image_dir    = data/gz2/images
    out_dir      = data/gz2/processed/splits/standard/
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from constants import (
    CLASS_NAMES,
    COL_SMOOTH, COL_FEATURED, COL_EDGEON, COL_NOT_EDGEON, COL_SPIRAL, COL_NOSPIRAL,
    DEFAULT_SEED
)

####                CONFIG                 ####
DEFAULT_LABELED_CSV = Path("data/gz2/processed/gz2_labeled.csv")
DEFAULT_EASY_CSV = Path("data/gz2/processed/splits/easy.csv")
DEFAULT_IMAGE_DIR = Path("data/gz2/images")
BASE_OUT_DIR = Path("data/gz2/processed/splits")
DEFAULT_SIZE = 50_000

# Starting with a 70/15/15 split
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15 / (1 - TEST_FRACTION)   # This is ~0.1765 of the remaining 85%

# Soft target columns names to be written to CSVs
SOFT_COLS = [f"soft_{k}" for k in range(4)]


####                FUNCTIONS               ####

def _resolve_out_dir(args) -> Path:
    """Derive appropriate output directory from flags when not explicitly provided

    --domain-split only     -> splits/domain/
    --soft-targets only     -> splits/soft/
    both                    -> splits/domain_soft/
    neither                 -> splits/full
    """
    
    # Respect user-provided output directory path
    if args.out_dir is not None:
        return Path(args.out_dir)
    
    if args.domain_split and args.soft_targets:
        return BASE_OUT_DIR / "domain_soft"
    if args.domain_split:
        return BASE_OUT_DIR / "domain"
    if args.soft_targets:
        return BASE_OUT_DIR / "soft"
    
    return BASE_OUT_DIR / "full"


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Subsample the input data for n rows, preserving per-class proportions

    Args:
        df (pd.DataFrame): labled DataFrame with a `gz2_class` column (see analysis.py)
        n (int): target number of rows
        seed (int): random seed

    Returns:
        pd.DataFrame: subsampled DataFrame
    """
    if n >= len(df):
        print(f"    Requested {n:,} rows, only {len(df):,} available. Keeping all rows.")
        return df.reset_index(drop=True)
    
    # Get per-class target count proportional to current class balance
    class_counts = df["gz2_class"].value_counts().sort_index()
    proportions = class_counts / class_counts.sum()
    per_class_target = (proportions * n).round().astype(int)
    
    diff = n - per_class_target.sum()
    
    # If rounding is off, truncate the most prevalent class
    if diff != 0:
        largest = per_class_target.idxmax()
        per_class_target[largest] += diff
        
    print(f"    Per-class subsample targets: {per_class_target.to_dict()}")
    
    pieces = []
    for k, target in per_class_target.items():
        class_df = df[df["gz2_class"] == k]
        pieces.append(class_df.sample(n=target, random_state=seed))
        
    out = pd.concat(pieces, ignore_index=True)
    
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return out


def compute_soft_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 4 soft targets from the decision tree vote fractions

    P(Elliptical) = p_smooth
    P(Edge-on disk) = p_featured * p_edgeon
    P(Face-on spiral) = p_featured * p_not_edgeon * p_spiral
    P(Face-on spiral) = p_featured * p_not_edgeon * p_no_spiral
    
    Four values are renormalized per-row to sum to 1.0

    Args:
        df (pd.DataFrame): DataFrame with 6 vote-fraction columns

    Returns:
        pd.DataFrame: DataFrame with four new `soft_k` columns
    """
    p_smooth = df[COL_SMOOTH].to_numpy()
    p_featured = df[COL_FEATURED].to_numpy()
    p_edgeon = df[COL_EDGEON].to_numpy()
    p_not_edge = df[COL_NOT_EDGEON].to_numpy()
    p_spiral = df[COL_SPIRAL].to_numpy()
    p_no_spiral = df[COL_NOSPIRAL].to_numpy()
    
    raw = np.column_stack([
        p_smooth,
        p_featured * p_edgeon,
        p_featured * p_not_edge * p_spiral,
        p_featured * p_not_edge * p_no_spiral
    ])
    
    # Renormalize
    row_sums = raw.sum(axis=1, keepdims=True)
    zero_mask = (row_sums.flatten() == 0)
    
    if zero_mask.any():
        print(f"    WARNING {zero_mask.sum()} rows have zero-sum soft targets;"
              f"falling back to one-hot from gz2_class"
            )
        
        raw[zero_mask] = np.eye(4)[df.loc[zero_mask, "gz2_class"].to_numpy()]
        row_sums[zero_mask] = 1.0
        
    normalized = raw / row_sums
    
    # Sanity check that argmax of soft target aligns with hard label
    soft_argmax = normalized.argmax(axis=1)
    agreement = (soft_argmax == df["gz2_class"].to_numpy()).mean()
    print(f"    Soft argmax vs hard label agreement: {100 * agreement:.2f}%")
    
    out = df.copy()
    for k in range(4):
        out[SOFT_COLS[k]] = normalized[:, k]
    
    return out


def stratified_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratifies a 70/15/15 train/val/test split on gz2_class

    Args:
        df (pd.DataFrame): subsampled DataFrame with gz2_class column
        seed (int): random seed

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    # Peel out test
    train_val, test = train_test_split(
        df, test_size=TEST_FRACTION,
        stratify=df["gz2_class"],
        random_state=seed
    )
    
    # Split train_val into train and val    
    train, val = train_test_split(
        train_val,
        test_size=VAL_FRACTION,
        stratify=train_val["gz2_class"],
        random_state=seed
    )
    
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # Report class composition
    print(f"    train: {len(train):,} val: {len(val):,} test: {len(test):,}")
    
    for name, part in [("train", train), ("val", val), ("test", test)]:
        counts = part["gz2_class"].value_counts().sort_index();
        percent = 100 * counts / len(part)
        dist = " ".join(f"{CLASS_NAMES[k]}: {counts[k]:,} ({percent[k]:.1f}%)" for k in counts.index)
        
        print(f"    {name}: {dist}")
        
    return train, val, test


def compute_norm_stats(
    train_df: pd.DataFrame, image_dir: Path, 
    sample_size: int | None = None, seed: int=DEFAULT_SEED
) -> dict[str, list[float]]:
    """Compute the per-channel RGB mean and standard dev from training images


    Args:
        train_df (pd.DataFrame): training split DataFrame with "asset_id"
        image_dir (Path): directory containing <asset_id>.jpg
        sample_size (int | None, optional): How many image sets used for statistics. Defaults to None.
        seed (int, optional): seed for random sampling. Defaults to DEFAULT_SEED.
        
    Returns:
        {
            "mean": list of means,
            "std": list of std deviations
        }
    """
    image_dir = Path(image_dir)
    
    if sample_size is not None and sample_size < len(train_df):
        sample_df = train_df.sample(n=sample_size, random_state=seed)
        print(f"    Using {sample_size:,} training images for stats")
    else:
        sample_df = train_df
        print(f"    Using all {len(sample_df):,} training images for stats")
        
    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    
    missing = 0
    bad = 0
    
    # The dataframe should have already been checked that images exist, but
    # this double checks to be safe. 
    for i, asset_id in enumerate(sample_df["asset_id"].astype(str)):
        path = image_dir / f"{asset_id}.jpg"
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            missing += 1
            continue
        
        # HxWx3 in [0,1]
        arr = np.asarray(img, dtype=np.float64) / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            bad += 1
            continue
        
        pixels = arr.shape[0] * arr.shape[1]
        n_pixels += pixels
        channel_sum += arr.sum(axis=(0,1))
        channel_sq_sum += (arr ** 2).sum(axis=(0,1))
        
        # Print message every 5000 images
        if (i + 1) % 5000 == 0:
            print(f"    processed {i + 1:,} / {len(sample_df):,}")
            
    if missing or bad:
        print(f"    Skipped {missing} missing and {bad} malformed images")
        
    mean = channel_sum / n_pixels
    var = (channel_sq_sum / n_pixels) - mean ** 2
    std = np.sqrt(np.maximum(var, 0))       # Guards against tiny negatives from floating point noise
    
    print(f"    Channel mean: {mean.round(4).tolist()}")
    print(f"    Channel std: {std.round(4).tolist()}")
    
    return {"mean": mean.tolist(), "std": std.tolist()}
        
        
def write_outputs(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  stats: dict, out_dir: Path) -> None:
    """Writes split CSVs and stats.json to out_dir

    Args:
        train (pd.DataFrame): training data
        val (pd.DataFrame): validation data
        test (pd.DataFrame): test data
        stats (dict): normalization statistics
        out_dir (Path): output directory
    """
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    
    (out_dir / "stats.json").write_text(json.dumps(stats, indent = 2))
    
    print(f"\n    Wrote {out_dir/'train.csv'}")
    print(f"    Wrote {out_dir/'val.csv'}")
    print(f"    Wrote {out_dir/'test.csv'}")
    print(f"    Wrote {out_dir/'stats.json'}")
    
    
def compute_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Computes the inverse-frequency class weights for WeightedRandomSampler

    Weights are normalized to sum to 1.0 so their scale is stable regardless of class
    distribution or count

    Args:
        train_df (pd.DataFrame): training split with gz2_class column

    Returns:
        np.ndarray: array with shape (N_CLASSES,), containing weights for each class
    """
    counts = train_df["gz2_class"].value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights /= weights.sum()
    
    print("    Per-class weights (normalized):")
    for k, (name, w) in enumerate(zip(CLASS_NAMES.values(), weights)):
        print(f"    {name:<22}  count={counts[k]:>7,}  weight={w:.6f}")
        
    return weights


def add_sample_weights(train_df: pd.DataFrame, class_weights: np.ndarray) -> pd.DataFrame:
    """Adds a `sample_weight` column to the training DataFrame for use by the weighted sampler

    Args:
        train_df (pd.DataFrame): training split with gz2_class column
        class_weights (np.ndarray): per-class weights from `compute_class_weights()`

    Returns:
        pd.DataFrame: copy of train_df with the new `sample_weight` column
    """
    out = train_df.copy()
    out["sample_weight"] = class_weights[train_df["gz2_class"].to_numpy()]
    return out
    
    
####                MAIN                ####
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labeled_csv", nargs="?", default=str(DEFAULT_LABELED_CSV))
    parser.add_argument("image_dir", nargs="?", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("out_dir", nargs="?", default=None)
    
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"target total sample size (default {DEFAULT_SIZE:,})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"target random seed (default {DEFAULT_SEED})")
    parser.add_argument("--stats-sample", type=int, default=10_000, help=f"number of training images used to compute normalization stats (default 10,000; use -1 for all)")
    parser.add_argument("--soft-targets", action="store_true", help="Compute and write branch-product soft targets")
    parser.add_argument("--domain-split", action="store_true", help="Use easy.csv instead of gz2_labeled.csv as input")
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    out_dir = _resolve_out_dir(args)
    stats_sample = None if args.stats_sample < 0 else args.stats_sample
    
    # Get input filepath
    if args.domain_split and args.labeled_csv == str(DEFAULT_LABELED_CSV):
        labeled_csv = DEFAULT_EASY_CSV
        print(f"    Domain-split enabled: loading default output from analysis.py ({labeled_csv})")
    elif args.domain_split:
        labeled_csv = Path(args.labeled_csv)
        print(f"    Domain-split with custom CSV enabled: loading {labeled_csv}")    
    else:
        labeled_csv = Path(args.labeled_csv)
        print(f"    Loading labeled CSV: {labeled_csv}")
        
    print(f"Output directory: {out_dir}\n")
    # Loading
    df = pd.read_csv(labeled_csv)
    print(f"    {len(df):,} rows loaded")
    
    bad = (df["gz2_class"] < 0) | (df["gz2_class"] > 3)
    
    if bad.any():
        print(f"    Dropping {bad.sum()} rows with invalid labels")
        df = df[~bad].reset_index(drop=True)

    # Subsample        
    print(f"\nStratified subsample to {args.size:,} rows (seed={args.seed})")
    df = stratified_subsample(df, args.size, args.seed)
    print(f"    Subsampled to {len(df):,} rows")
    
    # Get soft targets if requested
    if args.soft_targets:
        print("\nComputing branch-product soft targets:")
        df = compute_soft_targets(df)
    else:
        print("\nSkipping soft targets (pass --soft-targets to enable)")
        
    # Splits
    print(f"\nStratified 70/15/15 split (seed={args.seed})")
    train, val, test = stratified_split(df, args.seed)
    
    # Class and sample weights
    print("\nComputing class weights for WeightedRandomSampler")
    class_weights = compute_class_weights(train)
    train = add_sample_weights(train, class_weights)
    
    # Normalization stats
    print("\nComputing normalization statistics from training images:")
    stats = compute_norm_stats(train, image_dir, sample_size=stats_sample, seed=args.seed)
    
    stats_payload = {
        "normalization": stats,
        "splits": {
            "train_n": len(train),
            "val_n": len(val),
            "test_n": len(test),
        },
        "metadata": {
            "labeled_csv": str(labeled_csv),
            "image_dir": str(image_dir),
            "subsample_size": args.size,
            "seed": args.seed,
            "domain_split": args.domain_split,
            "stats_sample_size": stats_sample if stats_sample else "all",
            "soft_targets_computed": args.soft_targets,
            # In case I choose to use another method later
            "soft_target_construction": "branch_product_renormalized" if args.soft_targets else None    
        }
    }
    
    # Write
    print(f"\nWriting outputs to {out_dir}:")
    write_outputs(train, val, test, stats_payload, out_dir)
    
    print("\nDone.")
    
    
if __name__ == "__main__":
    main()
    