"""
Danyal Ahmed - April 2026

analysis.py
Exploration and analysis script for the distribution of GZ2
dataset for preprocessing.

Joins gz2_hart16.csv, gz2samples.csv, and gz2_filename_mapping.csv, 
and constructs the 4-label class counts from debiased vote fractions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import constants


###################### CONSTANTS ######################

####        Column names from gz2_hart16.csv       ####

# # T01 - Smooth vs features/disk
# COL_SMOOTH = "t01_smooth_or_features_a01_smooth_debiased"
# COL_FEATURED = "t01_smooth_or_features_a02_features_or_disk_debiased"

# # T02 - Edge-on disk
# COL_EDGEON = "t02_edgeon_a04_yes_debiased"
# COL_NOT_EDGEON = "t02_edgeon_a05_no_debiased"

# # T04 - Spiral arms
# COL_SPIRAL = "t04_spiral_a08_spiral_debiased"
# COL_NOSPIRAL = "t04_spiral_a09_no_spiral_debiased"

# ####        Join keys       ####
# HART_KEY = "dr7objid"       # gz2_hart16.csv
# SAMPLES_KEY = "OBJID"       # gz2samples.csv
# MAPPING_KEY = "objid"       # gz2_filename_mapping.csv

# ####        Default Thresholds      ####
# THRESHOLDS = {
#     "smooth": 0.7,
#     "edgeon": 0.7,
#     "spiral": 0.7
# }

# OOD_SPLIT_COL = "REDSHIFT"
# OOD_SPLIT_DEFAULT = 70      # percentile

# ####        Class Definitions       ####
# CLASS_NAMES = {
#     0: "Elliptical",
#     1: "Edge-on disk",
#     2: "Face-on spiral",
#     3: "Face-on non-spiral"
# }

# CLASS_SOFT_COL = {
#     0: COL_SMOOTH,
#     1: COL_EDGEON,
#     2: COL_SPIRAL,
#     3: COL_NOSPIRAL
# }

# CLASS_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


############################# FUNCTIONS #############################

####        Data Loading        ####
def load_hart(path: Path) -> pd.DataFrame:
    """Load gz2_hart16.csv and keep only the columns needed for labeling

    Args:
        path (Path): csv path
        
    Returns:
        pd.DataFrame: Pandas DataFrame containing rows with relevant columns
    """
    print(f"Loading Hart16 vote fractions: {path}")
    cols = [constants.HART_KEY, constants.COL_SMOOTH, constants.COL_FEATURED, 
            constants.COL_EDGEON, constants.COL_NOT_EDGEON, constants.COL_SPIRAL, constants.COL_NOSPIRAL]
    
    df = pd.read_csv(path, usecols=cols)
    print(f"    {len(df):,} rows loaded")
    return df


def load_samples(path: Path) -> pd.DataFrame:
    """
    Load gz2samples.csv, keeping redshift and size covariates for out-of-domain
    analysis.
    
    Pulls the following metadata columns:
        REDSHIFT    -   spectroscopic redshift (this is my primary domain-split axis)
        FRACDEV_R   -   de Vaucouleurs profile coefficient in the r-band, useful as
                        an independent photometric morphology proxy
        PETROR50_R  -   Petrosian 50% angular radius in arcseconds, in r-band
        PETROMAG_R  -   Petrosian apparent magnitude in r-band
        PETROMAG_MR -   Petrosian absolute magnitude in r-band (k-corrected)
        REGION      -   sub-sample region flag
    

    Args:
        path (Path): csv filepath

    Returns:
        pd.DataFrame: Pandas DataFrame containing all rows with relevant columns
    """
    print(f"Loading samples metadata: {path}")
    cols = [constants.SAMPLES_KEY, "REDSHIFT", "FRACDEV_R", "PETROR50_R", "PETROMAG_R", "PETROMAG_MR", "REGION"]
    
    df = pd.read_csv(path, usecols=cols)
    
    # Normalize join key to lowercase for merge
    df = df.rename(columns={constants.SAMPLES_KEY: constants.HART_KEY})
    print(f"    {len(df):,} rows loaded, columns: {cols}")
    return df
    
    
def load_mapping(path: Path) -> pd.DataFrame:
    """Load gz2_filename_mapping.csv (objid -> asset_id)

    Args:
        path (Path): csv filepath

    Returns:
        pd.DataFrame: Pandas DataFrame containing all mapping rows
    """
    print(f"Loading filename mapping: {path}")
    df = pd.read_csv(path)
    
    # Normalize join key to lowercase
    df = df.rename(columns={constants.MAPPING_KEY: constants.HART_KEY})
    print(f"    {len(df):,} rows loaded")
    return df


def build_master(hart_path: Path, samples_path: Path, mapping_path: Path) -> pd.DataFrame:
    """Join three data sources on object id

    Args:
        hart_path (Path): main data csv filepath
        samples_path (Path): metadata csv filepath
        mapping_path (Path): mapping csv filepath

    Returns:
        pd.DataFrame: Pandas DataFrame with all vote fractions, metadata, and asset ids
    """
    
    hart = load_hart(hart_path)
    samples = load_samples(samples_path)
    mapping = load_mapping(mapping_path)
    
    # Inner join
    df = hart.merge(samples, on=constants.HART_KEY, how="inner")
    df = df.merge(mapping, on=constants.HART_KEY, how="inner")
    
    print(f"\nMaster table: {len(df):,} rows after three-way join")
    print(f"    {len(hart) - len(df):,} rows dropped")
    return df


def audit_images(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Cross-reference labeled DataFrame against available images

    Args:
        df (pd.DataFrame): joined, unlabeled DataFrame
        image_dir (Path): directory containing <asset_id>.jpg files

    Returns:
        pd.DataFrame: filtered DataFrame containing only rows with associated images
    """
    image_dir = Path(image_dir)
    
    print(f"\nAuditing images in {image_dir}...")
    
    # I initially used .apply(), but this is quite slow
    # exists = df["asset_id"].apply(lambda asset_id: (image_dir/f"{asset_id}.jpg").exists())
    
    # Building the glob is a LOT faster, 15s -> <1s
    available = {p.stem for p in image_dir.glob("*.jpg")}
    exists = df["asset_id"].astype(str).isin(available)
    
    missing = (~exists).sum()   # Someone said they're unfamiliar with this syntax; 
                                # '~' is bitwise NOT, or element-wise with a DataFrame
                                
    print(f"    {exists.sum():,} images found, {missing:,} missing ({100*missing / len(df):.1f}%)")
    
    return df[exists].reset_index(drop=True)


def compute_ood_split(df: pd.DataFrame, labels: pd.Series,
                      percentile: int = constants.OOD_SPLIT_DEFAULT) -> tuple[float, pd.Series, pd.Series]:
    """Compute a redshift-based easy/hard domain split at the given percentile.

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        percentile (int): redshift percentile for the easy/hard boundary.
                          Default is 50 (median split).

    Returns:
        tuple:
            z_cut (float):          the redshift cutoff value
            easy_mask (pd.Series):  boolean mask for the easy (low-z) regime
            hard_mask (pd.Series):  boolean mask for the hard (high-z) regime
    """
    z     = df[constants.OOD_SPLIT_COL]
    z_cut = float(np.percentile(z.dropna(), percentile))

    easy_mask = (z <  z_cut) & (labels >= 0)
    hard_mask = (z >= z_cut) & (labels >= 0)

    return z_cut, easy_mask, hard_mask


####        Label Construction          ####

def assign_labels(df: pd.DataFrame, threshold: dict[str, float]) -> pd.Series:
    """Assigns 4 class labels using the GZ2 decision tree paths
    
    0   Elliptical              T01=smooth
    1   Edge-on disk            T01=featured    AND     T02=edge-on
    2   Face-on spiral          T01=featured    AND     T02=not-edge-on     AND     T04=spiral
    3   Face-on non-spiral      T01=featured    AND     T02=not-edge-on     AND     T04=no-spiral
    -1  Ambiguous
    
    NOTE: the 'featured' mask reuses the threshold['smooth'], since smooth vs. featured
    vote fractoins are complementary. I may split this into two keys later if I need
    to implement per-task threshold sweeps
    
    Args:
        df (pd.DataFrame): master joined DataFrame
        threshold (dict[str, float]): keys 'smooth', 'edgeon', and 'spiral
    
    Returns:
        pd.Series: 
    """
    
    thresh_smooth, thresh_edge, thresh_spiral = threshold["smooth"], threshold["edgeon"], threshold["spiral"]
    labels = pd.Series(-1, index=df.index, dtype=int)
    
    smooth = df[constants.COL_SMOOTH] >= thresh_smooth
    featured = df[constants.COL_FEATURED] >= thresh_smooth
    edgeon = df[constants.COL_EDGEON] >= thresh_edge
    not_edgeon = df[constants.COL_NOT_EDGEON] >= thresh_edge
    spiral = df[constants.COL_SPIRAL] >= thresh_spiral
    no_spiral = df[constants.COL_NOSPIRAL] >= thresh_spiral
    
    # ! Note that this logic assumes a minimum threshold of 0.5
    # ! With thresholds <0.5, reassignment falls through to 
    # ! the next line and higher-index classes will be over-represented
    labels[smooth] = 0
    labels[featured & edgeon] = 1
    labels[featured & not_edgeon & spiral] = 2
    labels[featured & not_edgeon & no_spiral] = 3
    
    return labels



####        Summary and Reporting Functions         ####

def print_summarize_vote_fractions(df: pd.DataFrame) -> None:
    """Print descriptive stats for the six key vote fraction columns

    Args:
        df (pd.DataFrame): main joined DataFrame
    """
    print("\nVote fraction summary statistics:")
    print(f"    {'Column':<55} {'mean':^6} {'std':^6} {'<0.3':^9} {'0.3-0.7':^9} {'>0.7':^9}")
    print(" " + "-" * 82)
    for col in [constants.COL_SMOOTH, constants.COL_FEATURED, constants.COL_EDGEON, 
                constants.COL_NOT_EDGEON, constants.COL_SPIRAL, constants.COL_NOSPIRAL]:
        s = df[col].dropna()
        low = (s < 0.3).sum()
        mid = ((s >= 0.3) & (s<=0.7)).sum()
        high = (s > 0.7).sum()
        
        print(f"    {col:<55} {s.mean():>6.3f} {s.std():>6.3f} {low:>9,} {mid:>9,} {high:>9,}")
        
        
def print_class_balance_report(labels: pd.Series, title: str="Class balance") -> int:
    """Print the class counts and percentages

    Args:
        labels (pd.Series): label Series
        title (str, optional): Title to print in output. Defaults to "Class balance".
        
    Returns:
        int: total number of valid, labeled rows
    """
    total = (labels >= 0).sum()
    excluded = (labels < 0).sum()
    
    print(f"\n{title}")
    print(f"    Retrained: {total:,}    Excluded: {excluded:,} ({100 * excluded / (total + excluded):.1f}%)")
    
    for k, name in constants.CLASS_NAMES.items():
        n = (labels == k).sum()
        print(f"    {name:<22} {n:>8}   {100*n/total:>5.1f}%")
        
    return total


def print_threshold_sweep(df: pd.DataFrame) -> None:
    """ Prints the effects of varying shared thresholds on class counts and total class sizes

    Args:
        df (pd.DataFrame): joined dataset
    """
    
    print("\nThreshold sweep (shared across all tasks):")
    header = f"  {'Thresh':>7}  {'Elliptical':^8}  {'EdgeOn':^8}  {'Spiral':^8}  {'Non-Spir':^8}  {'Total':^8}  {'Excluded':>6}"
    print(header)
    
    print(" " + "-" * 72)
    
    # Sweep from 0.5 to 0.9 in 0.05 increments
    for thresh in np.arange(0.50, 0.91, 0.05):
        lbl = assign_labels(df, {"smooth": thresh, "edgeon": thresh, "spiral": thresh})
        counts = [(lbl == k).sum() for k in range(4)]
        total = sum(counts)
        excld = 100 * (lbl < 0).sum() / len(lbl)
        row = f"  {thresh:>7.2f}  " + "  ".join(f"{c:>8,}" for c in counts)
        print(f"{row}   {total:>8}  {excld:>6.1f}%")
    
    
def print_covariate_summary(df: pd.DataFrame, labels: pd.Series) -> None:
    """Prints per-class descriptive statistics for REDSHIFT and FRACDEV_R

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned with df.index
    """
    
    covariates = [
        ("REDSHIFT", "Redshift"),
        ("FRACDEV_R", "FRACDEV_R (1=bulge, 0=disk)"),
    ]
    
    for col, label in covariates:
        if col not in df.columns:
            print(f"\nSkipping {col} summary (column not found)")
            continue
        
        print(f"\nPer-class {label} summary:")
        print(f"    {'Class':<22} {'n':>7}  {'mean':>6}  {'std':>6}  {'p10':>6}  {'p25':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}")
        print("   " + "-" * 78)
        
        for k, name in constants.CLASS_NAMES.items():
            s = df.loc[labels == k, col].dropna()
            p = np.percentile(s, [10, 25, 50, 75, 90])
            print(f"    {name:<22} {len(s):>7}  {s.mean():>6.3f}  {s.std():>6.3f}  "
                  f"{p[0]:>6.3f}  {p[1]:>6.3f}  {p[2]:>6.3f}  {p[3]:>6.3f}  {p[4]:>6.3f}")
            
            
def print_domain_split_sweep(df: pd.DataFrame, labels: pd.Series) -> None:
    """
    Print redshift cutoff sweeps and report per-class counts and mean FRACDEV_R 
    in easy (low-z) and hard (high-z) regimes

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
    """
    if "REDSHIFT" not in df.columns or "FRACDEV_R" not in df.columns:
        print("\nSkipping domain split sweep (REDSHIFT or FRACDEV_R column(s) not found)")
        return
    
    # Sweep from 10th to 90th percentile in 10-percentile steps
    z = df.loc[labels >= 0, "REDSHIFT"].dropna()
    cutoffs = np.percentile(z, np.arange(10, 91, 10))
    
    print("\nDomain split sweep (easy: z < cutoff, hard: z >= cutoff):")
    print(f"  {'z*':>6}  {'regime':<6}  " + "  ".join(f"{constants.CLASS_NAMES[k][:10]:^12}" for k in range(4)) +
          f"  {'total':^8}  {'mean FRACDEV_R':^14}")
    print("  " + "-" * 100)
    
    for cutoff in cutoffs:
        for regime, mask_func in [
            ("easy", lambda z: z < cutoff),
            ("hard", lambda z: z >= cutoff)
        ]:
            regime_mask = mask_func(df["REDSHIFT"]) & (labels >= 0)
            counts = [(regime_mask & (labels == k)).sum() for k in range(4)]
            total = sum(counts)
            fracdev_mean = df.loc[regime_mask, "FRACDEV_R"].mean()
            row = f"  {cutoff:>6.4f}  {regime:<6}  " + "  ".join(f"{c:>12,}" for c in counts) + f"  {total:>8,}  {fracdev_mean:>14.3f}"
            print(row)
            
        print()
        
        
def print_angular_size_summary(df: pd.DataFrame, labels: pd.Series) -> None:
    """Print per-class PETROR50_R descriptive statistics

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
    """
    col = "PETROR50_R"
    if col not in df.columns:
        print(f"\nSkipping angular size summary ({col} not found)")
        return
    
    print(f"\nPer-class angular size ({col}, arcsec) summary:")
    print(f"    {'Class':<22} {'n':>7}  {'mean':>6}  {'std':>6}  "
          f"{'p10':>6}  {'p50':>6}  {'p90':>6}  {'>15\"':>7}  {'>20\"':>7}")
    print("   " + "-" * 86)

    for k, name in constants.CLASS_NAMES.items():
        s = df.loc[labels == k, col].dropna()
        p = np.percentile(s, [10, 50, 90])
        pct_15 = 100 * (s > 15).sum() / len(s)
        pct_20 = 100 * (s > 20).sum() / len(s)
        print(f"    {name:<22} {len(s):>7,}  {s.mean():>6.2f}  {s.std():>6.2f}  "
              f"{p[0]:>6.2f}  {p[1]:>6.2f}  {p[2]:>6.2f}  "
              f"{pct_15:>6.1f}%  {pct_20:>6.1f}%")
        
        
def print_ood_split_report(df: pd.DataFrame, labels: pd.Series,
                           percentile: int = constants.OOD_SPLIT_DEFAULT) -> float:
    """Print class balance and covariate stats for the chosen OOD split.

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        percentile (int): redshift percentile cutoff. Defaults to 50.

    Returns:
        float: the redshift cutoff value (for use downstream)
    """
    z_cut, easy_mask, hard_mask = compute_ood_split(df, labels, percentile)

    print(f"\nOOD split at z = {z_cut:.4f} (p{percentile} of labeled set)")
    print(f"    {'Class':<22} {'easy N':>8}  {'easy %':>7}  {'hard N':>8}  {'hard %':>7}  {'ratio':>7}")
    print("   " + "-" * 68)

    for k, name in constants.CLASS_NAMES.items():
        n_easy = (easy_mask & (labels == k)).sum()
        n_hard = (hard_mask & (labels == k)).sum()
        n_tot  = (labels == k).sum()
        ratio  = n_easy / n_hard if n_hard > 0 else float("inf")
        print(f"    {name:<22} {n_easy:>8,}  {100*n_easy/n_tot:>6.1f}%  "
              f"{n_hard:>8,}  {100*n_hard/n_tot:>6.1f}%  {ratio:>7.2f}x")

    # Summary covariates per regime
    for regime, mask in [("easy", easy_mask), ("hard", hard_mask)]:
        z_s = df.loc[mask, constants.OOD_SPLIT_COL]
        f_s = df.loc[mask, "FRACDEV_R"] if "FRACDEV_R" in df.columns else None
        frac_str = f"  mean FRACDEV_R = {f_s.mean():.3f}" if f_s is not None else ""
        print(f"    {regime:<6}  n = {mask.sum():>7,}  "
              f"z: [{z_s.min():.4f}, {z_s.max():.4f}]  mean z = {z_s.mean():.4f}{frac_str}")

    return z_cut

####        Plotting functions        ####

def plot_vote_distributions(df: pd.DataFrame, thresh: dict[str, float], save: bool=False) -> None:
    """Creates histograms of the three primary vote fractions with threshold lines.

    Args:
        df (pd.DataFrame): joined dataset
        thresh (dict[str, float]): dict includign 'smooth', 'edgeon' and 'spiral' thresholds
        save (bool, optional): whether to save the plot or simply show it. Defaults to False.
    """
    pairs = [
        (constants.COL_SMOOTH, thresh['smooth'], "T01: Smooth fraction"),
        (constants.COL_EDGEON, thresh['edgeon'], "T02: Edge-on fraction"),
        (constants.COL_SPIRAL, thresh['spiral'], "T04: Spiral fraction")
    ]
    
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, thr, title) in zip(axes, pairs):
        vals = df[col].dropna()
        ax.hist(vals, bins=60, color="#4878CF", edgecolor="none", alpha=0.85)
        ax.axvline(thr, color="crimson", lw=1.5, ls="--", label=f"Threshold = {thr:.2f}")
        ax.axvline(1 - thr, color="orange", lw=1.5, ls="--", label=f"Opposite = {1-thr:.2f}")
        ax.set_title(title)
        ax.set_xlabel("Debiased vote fraction")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        
    fig.suptitle("Vote Fraction Distributions", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/vote_distributions.png", save)
    
    
def plot_class_balance(labels: pd.Series, save=False) -> None:
    """Creates a bar chart of per-class sample counts

    Args:
        labels (pd.Series): label Series corresponding to data indices
        save (bool, optional): whether to save the plot instead of show. Defaults to False.
    """
    counts = [(labels == k).sum() for k in range(4)]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(constants.CLASS_NAMES.values(), counts, color=constants.CLASS_COLORS, edgecolor=None)
    
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + max(counts) * 0.01,
                f"{n:,}", ha='center', va='bottom', fontsize=9
        )
        
    ax.set_title("Class Balance after Thresholding")
    ax.set_ylabel("Sample count")
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/class_balance.png", save)
        
        
def plot_threshold_sweep(df: pd.DataFrame, save=False) -> None:
    """Creates line plots of per-class and total counts vs. a shared threshold

    Args:
        df (pd.DataFrame): joined dataset with labels
        save (bool, optional): whether to save the plot instead of showing. Defaults to False.
    """
    
    thresholds = np.arange(0.5, 0.91, 0.025)
    counts = {k: [] for k in range(4)}
    totals = []
    for thresh in thresholds:
        label = assign_labels(df, {"smooth": thresh, "edgeon": thresh, "spiral": thresh})
        for k in range(4):
            counts[k].append((label == k).sum())
        totals.append((label >= 0).sum())
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
    for k, color in zip(range(4), constants.CLASS_COLORS):
        ax1.plot(thresholds, counts[k], label=constants.CLASS_NAMES[k], color=color, lw=2)
    
    ax1.axvline(constants.THRESHOLDS['smooth'], color="gray", lw=1, ls=":", label=f"Default ({constants.THRESHOLDS['smooth']})")
    
    ax1.set_title("Per-Class Counts vs. Threshold")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    
    ax2.plot(thresholds, totals, color="steelblue", lw=2)
    ax2.axvline(constants.THRESHOLDS["smooth"], color='gray', lw=1, ls=":")
    ax2.set_title("Total Retained Samples vs. Threshold")
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("Sampled retained")
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/threshold_sweep.png", save)
    
    
def plot_redshift_distribution(df: pd.DataFrame, labels: pd.Series, save=False) -> None:
    """Create per-class redshift distribution plots.
    
    Only plotted if REDSHIFT column is present in df (after joining to metadata)

    Args:
        df (pd.DataFrame): joined DataFrame
        labels (pd.Series): labels with index matching df
        save (bool, optional): whether to save plot instead of showing. Defaults to False.
    """
    if "REDSHIFT" not in df.columns:
        print("     Skipping redshift plot (REDSHIFT column not found)")
        return
    
    fig, ax = plt.subplots(figsize=(9, 4))
    
    for k, color in zip(range(4), constants.CLASS_COLORS):
        z = df.loc[labels==k, "REDSHIFT"].dropna()
        ax.hist(z, bins=60, color=color, alpha=0.6, 
                label=f"{constants.CLASS_NAMES[k]} (n={len(z):,})", edgecolor=None
        )
        
    ax.set_title("Redshift Distribution by Class")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/redshift_by_class.png", save)
    
    
def plot_fracdev_distribution(df: pd.DataFrame, labels: pd.Series, save: bool = False) -> None:
    """Create per-class FRACDEV_R distribution histograms
    
    Generally, one can expect:
        Ellipticals concentrated near 1.0
        Edge-on disks concentrated near 0.0 (maybe a high-end tail from bulge-dominated edge-on views)
        Face-on spirals concentrated near 0.0
        Face-on non-spirals more spread in the range [0,1]

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        save (bool, optional): whether to save to file instead of showing. Defaults to False.
    """
    if "FRACDEV_R" not in df.columns:
        print("    Skipping FRACDEV_R plot (FRACDEV_R column not found)")
        return
    
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, color in zip(range(4), constants.CLASS_COLORS):
        f = df.loc[labels == k, "FRACDEV_R"].dropna()
        ax.hist(f, bins=60, color=color, alpha=0.6, label=f"{constants.CLASS_NAMES[k]} (n={len(f):,})", edgecolor=None)
        
    ax.set_title("FRACDEV_R Distribution by Class")
    ax.set_xlabel("FRACDEV_R (1 = de Vaucouleurs / bulge), 0 = exponential / disk")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/fracdev_by_class.png", save)
    
    
def plot_redshift_fracdev_scatter(df: pd.DataFrame, labels: pd.Series, save: bool = False) -> None:
    """Create 2D scallter of REDSHIFT vs FRACDEV_R; one panel per class

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        save (bool, optional): whether to save to file instead of showing. Defaults to False.
    """
    if "REDSHIFT" not in df.columns or "FRACDEV_R" not in df.columns:
        print("    Skipping redshift/FRACDEV_R scatter (one or both columns not found)")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    
    for ax, (k, color) in zip(axes, zip(range(4), constants.CLASS_COLORS)):
        mask = labels == k
        x = df.loc[mask, "REDSHIFT"].dropna()
        y = df.loc[mask, "FRACDEV_R"].reindex(x.index)
        
        hexbin = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.1)
        ax.set_title(constants.CLASS_NAMES[k], fontsize=9)
        ax.set_xlabel("Redshift")
        if k == 0:
            ax.set_ylabel("FRACDEV_R")
            
        fig.colorbar(hexbin, ax=ax, label="count")
        
    fig.suptitle("Redshift vs. FRACDEV_R by Class", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/redshift_fracdev_scatter.png", save)
    
    
def plot_angular_size_distribution(df:pd.DataFrame, labels: pd.Series, save: bool = False) -> None:
    """Create overlapping per-class PETROR50_R histograms with a crop line

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        save (bool, optional): whether to save instead of show. Defaults to False.
    """
    col = "PETROR50_R"
    if col not in df.columns:
        print(f"    Skipping angular size plot ({col} not found)")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
    
    for k, color in zip(range(4), constants.CLASS_COLORS):
        s = df.loc[labels == k, col].dropna()
        ax1.hist(s, bins=80, color=color, alpha=0.55, label=f"{constants.CLASS_NAMES[k]} (n={len(s):,})", edgecolor=None)
        
    ax1.axvline(15, color="crimson", lw=1.5, ls="--", label="15\" boundary")
    ax1.axvline(20, color="darkorange", lw=1.5, ls="--", label="20\" boundary")
    ax1.set_title("PETROR50_R distribution by class")
    ax1.set_xlabel("PRETRO50_R (arcsec)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
       
    for k, color in zip(range(4), constants.CLASS_COLORS):
        s = df.loc[labels == k, col].dropna().sort_values()
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax2.plot(s.values, cdf, color=color, lw=1.5, label=constants.CLASS_NAMES[k])

    ax2.axvline(15, color="crimson",    lw=1.5, ls="--", label="15\"")
    ax2.axvline(20, color="darkorange", lw=1.5, ls="--", label="20\"")
    ax2.set_title("PETROR50_R CDF by class")
    ax2.set_xlabel("PETROR50_R (arcsec)")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend(fontsize=8)

    fig.suptitle("Angular Size (PETROR50_R) by Class", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/angular_size_by_class.png", save)
    

def plot_soft_label_distributions(df: pd.DataFrame, labels: pd.Series,
                                  thresh: dict, save: bool = False) -> None:
    """For each class, plot the distribution of the vote fraction that defined it.

    A spike near 1.0 means most examples are unambiguous — soft labels carry
    little information for those rows. A spread down toward the threshold means
    genuine uncertainty is present and soft labels are meaningful.

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        thresh (dict): thresholds dict (used to draw the lower bound line)
        save (bool, optional): save instead of show. Defaults to False.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, (k, color) in zip(axes, zip(range(4), constants.CLASS_COLORS)):
        col  = constants.CLASS_SOFT_COL[k]
        thr  = list(thresh.values())[min(k, 2)]   # smooth/edgeon/spiral threshold
        vals = df.loc[labels == k, col].dropna()

        # Fraction in the "ambiguous" zone [threshold, threshold + 0.15]
        ambig_pct = 100 * ((vals >= thr) & (vals < thr + 0.15)).sum() / len(vals)

        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor=None)
        ax.axvline(thr, color="crimson", lw=1.5, ls="--",
                   label=f"Threshold = {thr:.2f}")
        ax.set_title(f"{constants.CLASS_NAMES[k]}\n(n={len(vals):,})", fontsize=9)
        ax.set_xlabel("Vote fraction", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)
        ax.text(0.05, 0.92, f"Near-threshold: {ambig_pct:.1f}%",
                transform=ax.transAxes, fontsize=7,
                color="crimson", va="top")

    fig.suptitle("Soft Label (Vote Fraction) Distribution Within Each Class", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/soft_label_distributions.png", save)
   
   
def plot_sample_image_grid(df: pd.DataFrame, labels: pd.Series,
                           image_dir: Path, n_per_class: int = 8,
                           save: bool = False) -> None:
    """Display a random sample of images for each class in a grid.

    Layout: one row per class, n_per_class columns.
    Each image is shown at its native 424x424 resolution — this is a visual
    sanity check, not a preprocessed view.

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame (must contain asset_id)
        labels (pd.Series): label Series aligned to df.index
        image_dir (Path): directory containing <asset_id>.jpg files
        n_per_class (int, optional): images per class row. Defaults to 8.
        save (bool, optional): save instead of show. Defaults to False.
    """
    image_dir = Path(image_dir)
    n_classes = len(constants.CLASS_NAMES)
    rng = np.random.default_rng(seed=42)    # fixed seed for reproducibility

    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2, n_classes * 2 + 0.5))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for k, name in constants.CLASS_NAMES.items():
        class_df = df[labels == k]

        # Sample min(n_per_class, available) rows
        n_sample = min(n_per_class, len(class_df))
        sample   = class_df.sample(n=n_sample, random_state=42)

        for col_idx in range(n_per_class):
            ax = axes[k][col_idx]
            ax.axis("off")

            if col_idx >= n_sample:
                continue

            row       = sample.iloc[col_idx]
            img_path  = image_dir / f"{row['asset_id']}.jpg"

            try:
                img = Image.open(img_path)
                ax.imshow(np.array(img), cmap="gray" if img.mode == "L" else None)
            except (FileNotFoundError, OSError) as e:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)

            # First column: class label on the left
            if col_idx == 0:
                ax.text(-0.05, 0.5, name, transform = ax.transAxes, fontsize=9, rotation=90, ha="right", va="center")

            # Top row: column index header
            if k == 0:
                ax.set_title(f"#{col_idx + 1}", fontsize=8)

    fig.suptitle("Random Sample Images by Class (native 424x424)", y=1.01)
    _save_or_show(fig, "output/preprocessing/plots/sample_image_grid.png", save)
    
    
def plot_ood_split(df: pd.DataFrame, labels: pd.Series,
                  percentile: int = constants.OOD_SPLIT_DEFAULT,
                  save: bool = False) -> None:
    """Visualise the chosen OOD split across classes and covariates.

    Three panels:
        Left:   Redshift CDF with the split line and per-class curves
        Centre: Class composition of easy vs hard regimes (stacked bar)
        Right:  FRACDEV_R distribution in easy vs hard (overlapping histograms)

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        percentile (int): redshift percentile cutoff. Defaults to 50.
        save (bool, optional): save instead of show. Defaults to False.
    """
    z_cut, easy_mask, hard_mask = compute_ood_split(df, labels, percentile)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    #    Left: redshift CDFs                                                 
    for k, color in zip(range(4), constants.CLASS_COLORS):
        z = df.loc[labels == k, constants.OOD_SPLIT_COL].dropna().sort_values()
        cdf = np.arange(1, len(z) + 1) / len(z)
        ax1.plot(z.values, cdf, color=color, lw=1.5, label=constants.CLASS_NAMES[k])

    ax1.axvline(z_cut, color="black", lw=1.5, ls="--",
                label=f"p{percentile} cutoff z={z_cut:.4f}")
    ax1.set_title("Redshift CDF by class")
    ax1.set_xlabel("Redshift")
    ax1.set_ylabel("Cumulative fraction")
    ax1.legend(fontsize=7)

    #    Centre: class composition stacked bar                              
    regimes     = ["easy", "hard"]
    regime_masks = [easy_mask, hard_mask]
    bottoms      = np.zeros(2)

    for k, color in zip(range(4), constants.CLASS_COLORS):
        vals = [(m & (labels == k)).sum() for m in regime_masks]
        totals = [m.sum() for m in regime_masks]
        pcts = [100 * v / t for v, t in zip(vals, totals)]
        bars = ax2.bar(regimes, pcts, bottom=bottoms, color=color,
                       label=constants.CLASS_NAMES[k], edgecolor="white", linewidth=0.5)
        for bar, pct, bot in zip(bars, pcts, bottoms):
            if pct > 4:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bot + pct / 2, f"{pct:.1f}%",
                         ha="center", va="center", fontsize=7, color="white")
        bottoms += pcts

    ax2.set_title("Class composition by regime")
    ax2.set_ylabel("% of regime")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=7, loc="lower right")

    #    Right: FRACDEV_R in easy vs hard                                   
    if "FRACDEV_R" in df.columns:
        for mask, label, color, ls in [
            (easy_mask, f"easy (z < {z_cut:.3f})", "steelblue", "-"),
            (hard_mask, f"hard (z ≥ {z_cut:.3f})", "crimson",   "--"),
        ]:
            f = df.loc[mask, "FRACDEV_R"].dropna()
            ax3.hist(f, bins=60, color=color, alpha=0.5,
                     label=f"{label}\nmean={f.mean():.3f}", ls=ls,
                     edgecolor=None, density=True)
        ax3.set_title("FRACDEV_R: easy vs hard")
        ax3.set_xlabel("FRACDEV_R")
        ax3.set_ylabel("Density")
        ax3.legend(fontsize=7)
    else:
        ax3.set_visible(False)

    fig.suptitle(f"OOD Domain Split at z = {z_cut:.4f} (p{percentile})", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/ood_split.png", save)
     
    
def _save_or_show(fig: plt.Figure, filename: Path, save: bool):
    """PRIVATE FUNCTION - do not use outside of this file
    
    Either saves or shows a given Figure

    Args:
        fig (plt.Figure): figure to be shown or saved
        filename (Path): filepath if saving
        save (bool): whether to save (shows instead)
    """
    
    if save:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"    Saved {filename}")
    else:
        plt.show()
        
    plt.close(fig)
    
    
####            Export handler(s)           ####

def export_labeled_csv(df: pd.DataFrame, labels: pd.Series, out_path: Path) -> None:
    """Append class labels to master DataFrame and export retained rows.
    
    The asset_id column from the mapping file allows direct image lookup

    Args:
        df (pd.DataFrame): master joined DataFrame
        labels (pd.Series): Series of labels with matching index to df
        out_path (Path): where to save the new csv
    """
    out = df.copy()
    out["gz2_class"] = labels
    out["gz2_class_name"] = labels.map(constants.CLASS_NAMES).fillna("excluded")
    
    retained = out
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    retained.to_csv(out_path, index=False)
    
    print(f"\nSaved labeled CSV: {out_path}  ({len(retained):,} rows)")
    print(f"Columns: {list(retained.columns)}")
    
    
def export_split_csv(df: pd.DataFrame, labels: pd.Series,
                     z_cut: float, out_dir: Path) -> None:
    """Export separate easy and hard CSVs with a 'split' column added.

    Also writes the cutoff value to a small metadata text file so it is
    recorded alongside the data and doesn't need to be hardcoded downstream.

    Args:
        df (pd.DataFrame): labeled and filtered DataFrame
        labels (pd.Series): label Series aligned to df.index
        z_cut (float): redshift cutoff from compute_ood_split
        out_dir (Path): directory to write easy.csv, hard.csv, split_meta.txt
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["gz2_class"]      = labels
    out["gz2_class_name"] = labels.map(constants.CLASS_NAMES)
    out["split"]          = np.where(df[constants.OOD_SPLIT_COL] < z_cut, "easy", "hard")

    easy = out[out["split"] == "easy"].reset_index(drop=True)
    hard = out[out["split"] == "hard"].reset_index(drop=True)

    easy.to_csv(out_dir / "easy.csv", index=False)
    hard.to_csv(out_dir / "hard.csv", index=False)

    meta = (f"ood_split_col = {constants.OOD_SPLIT_COL}\n"
            f"ood_split_z_cut = {z_cut:.6f}\n"
            f"easy_n = {len(easy)}\n"
            f"hard_n = {len(hard)}\n")
    (out_dir / "split_meta.txt").write_text(meta)

    print(f"\n    easy.csv → {len(easy):,} rows")
    print(f"    hard.csv → {len(hard):,} rows")
    print(f"    split_meta.txt → z_cut = {z_cut:.6f}")

    
    
##################################### MAIN #####################################

def main(argv):
    """
    Usage:
        python analysis.py <gz2_hart16.csv> <gz2samples.csv> <mapping.csv> [out.csv] [--save]
        
    Default locations for saved files are:
        `output/preprocessing/plots/*.png`      for plots (if using --saved)
        `data/gz2/processed/gz2_labeled.csv`    for new labeled, filtered, and joined master CSV

    Example:
        python analysis.py data/gz2/raw/gz2_hart16.csv data/gz2/raw/gz2sample.csv data/gz2/raw/gz2_filename_mapping.csv labeled.csv --save
        
    Data flow:
        1. Build master table (three-way join)
        2. Audit images (drop rows with no associated image file)
        3. Assign labels (drop ambiguous rows and use threshold to add class columns)
        4. Report / plot / export on final set
    """
    if len(argv) < 4:
        print(main.__doc__)
        sys.exit(1)

    hart_path    = argv[1]
    samples_path = argv[2]
    mapping_path = argv[3]
    out_path     = (argv[4] if len(argv) > 4 and not argv[4].startswith("--")
                    else "data/gz2/processed/gz2_labeled.csv")
    save_figs    = "--save" in argv

    #    Build and audit                                                    
    df = build_master(hart_path, samples_path, mapping_path)
    df = audit_images(df, "data/gz2/images")

    #    Summaries on full audited set                                      
    print_summarize_vote_fractions(df)
    print_threshold_sweep(df)

    #    Assign and filter to labeled rows                                  
    print(f"\nAssigning labels at thresholds: {constants.THRESHOLDS}")
    labels_all = assign_labels(df, constants.THRESHOLDS)
    print_class_balance_report(labels_all, title="Class balance at default thresholds")

    df     = df[labels_all >= 0].reset_index(drop=True)
    labels = assign_labels(df, constants.THRESHOLDS)

    #    Covariate and domain analysis                                      
    print_covariate_summary(df, labels)
    print_angular_size_summary(df, labels)
    print_domain_split_sweep(df, labels)

    #    OOD split decision                                                 
    z_cut = print_ood_split_report(df, labels, percentile=constants.OOD_SPLIT_DEFAULT)

    #    Plots                                                              
    print(f"\n{'Saving' if save_figs else 'Generating'} plots...")
    plot_vote_distributions(df, constants.THRESHOLDS, save=save_figs)
    plot_class_balance(labels, save=save_figs)
    plot_threshold_sweep(df, save=save_figs)
    plot_redshift_distribution(df, labels, save=save_figs)
    plot_fracdev_distribution(df, labels, save=save_figs)
    plot_redshift_fracdev_scatter(df, labels, save=save_figs)
    plot_angular_size_distribution(df, labels, save=save_figs)
    plot_soft_label_distributions(df, labels, constants.THRESHOLDS, save=save_figs)
    plot_sample_image_grid(df, labels, "data/gz2/images", save=save_figs)
    plot_ood_split(df, labels, percentile=constants.OOD_SPLIT_DEFAULT, save=save_figs)

    #    Export                                                             
    export_labeled_csv(df, labels, out_path)
    export_split_csv(df, labels, z_cut, out_dir="data/gz2/processed/splits")
    
    
if __name__ == "__main__":
    main(sys.argv)
