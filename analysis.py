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


###################### CONSTANTS ######################

####        Column names from gz2_hart16.csv       ####

# T01 - Smooth vs features/disk
COL_SMOOTH = "t01_smooth_or_features_a01_smooth_debiased"
COL_FEATURED = "t01_smooth_or_features_a02_features_or_disk_debiased"

# T02 - Edge-on disk
COL_EDGEON = "t02_edgeon_a04_yes_debiased"
COL_NOT_EDGEON = "t02_edgeon_a05_no_debiased"

# T04 - Spiral arms
COL_SPIRAL = "t04_spiral_a08_spiral_debiased"
COL_NOSPIRAL = "t04_spiral_a09_no_spiral_debiased"

####        Join keys       ####
HART_KEY = "dr7objid"       # gz2_hart16.csv
SAMPLES_KEY = "OBJID"       # gz2samples.csv
MAPPING_KEY = "objid"       # gz2_filename_mapping.csv

####        Default Thresholds      ####
THRESHOLDS = {
    "smooth": 0.7,
    "edgeon": 0.7,
    "spiral": 0.7
}

####        Class Definitions       ####
CLASS_NAMES = {
    0: "Elliptical",
    1: "Edge-on disk",
    2: "Face-on spiral",
    3: "Face-on non-spiral"
}

CLASS_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


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
    cols = [HART_KEY, COL_SMOOTH, COL_FEATURED, COL_EDGEON, COL_NOT_EDGEON, COL_SPIRAL, COL_NOSPIRAL]
    
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
    cols = [SAMPLES_KEY, "REDSHIFT", "FRACDEV_R", "PETROR50_R", "PETROMAG_R", "PETROMAG_MR", "REGION"]
    
    df = pd.read_csv(path, usecols=cols)
    
    # Normalize join key to lowercase for merge
    df = df.rename(columns={SAMPLES_KEY: HART_KEY})
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
    df = df.rename(columns={MAPPING_KEY: HART_KEY})
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
    df = hart.merge(samples, on=HART_KEY, how="inner")
    df = df.merge(mapping, on=HART_KEY, how="inner")
    
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


####        Label Construction          ####

def assign_labels(df: pd.DataFrame, threshold: dict[str: float]) -> pd.Series:
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
    
    smooth = df[COL_SMOOTH] >= thresh_smooth
    featured = df[COL_FEATURED] >= thresh_smooth
    edgeon = df[COL_EDGEON] >= thresh_edge
    not_edgeon = df[COL_NOT_EDGEON] >= thresh_edge
    spiral = df[COL_SPIRAL] >= thresh_spiral
    no_spiral = df[COL_NOSPIRAL] >= thresh_spiral
    
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
    for col in [COL_SMOOTH, COL_FEATURED, COL_EDGEON, COL_NOT_EDGEON, COL_SPIRAL, COL_NOSPIRAL]:
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
    
    for k, name in CLASS_NAMES.items():
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
        
        for k, name in CLASS_NAMES.items():
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
    print(f"  {'z*':>6}  {'regime':<6}  " + "  ".join(f"{CLASS_NAMES[k][:10]:^12}" for k in range(4)) +
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
    

####        Plotting functions        ####

def plot_vote_distributions(df: pd.DataFrame, thresh: dict[str, float], save: bool=False) -> None:
    """Creates histograms of the three primary vote fractions with threshold lines.

    Args:
        df (pd.DataFrame): joined dataset
        thresh (dict[str, float]): dict includign 'smooth', 'edgeon' and 'spiral' thresholds
        save (bool, optional): whether to save the plot or simply show it. Defaults to False.
    """
    pairs = [
        (COL_SMOOTH, thresh['smooth'], "T01: Smooth fraction"),
        (COL_EDGEON, thresh['edgeon'], "T02: Edge-on fraction"),
        (COL_SPIRAL, thresh['spiral'], "T04: Spiral fraction")
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
    bars = ax.bar(CLASS_NAMES.values(), counts, color=CLASS_COLORS, edgecolor=None)
    
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
    for k, color in zip(range(4), CLASS_COLORS):
        ax1.plot(thresholds, counts[k], label=CLASS_NAMES[k], color=color, lw=2)
    
    ax1.axvline(THRESHOLDS['smooth'], color="gray", lw=1, ls=":", label=f"Default ({THRESHOLDS['smooth']})")
    
    ax1.set_title("Per-Class Counts vs. Threshold")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    
    ax2.plot(thresholds, totals, color="steelblue", lw=2)
    ax2.axvline(THRESHOLDS["smooth"], color='gray', lw=1, ls=":")
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
    
    for k, color in zip(range(4), CLASS_COLORS):
        z = df.loc[labels==k, "REDSHIFT"].dropna()
        ax.hist(z, bins=60, color=color, alpha=0.6, 
                label=f"{CLASS_NAMES[k]} (n={len(z):,})", edgecolor=None
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
    for k, color in zip(range(4), CLASS_COLORS):
        f = df.loc[labels == k, "FRACDEV_R"].dropna()
        ax.hist(f, bins=60, color=color, alpha=0.6, label=f"{CLASS_NAMES[k]} (n={len(f):,})", edgecolor=None)
        
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
    
    for ax, (k, color) in zip(axes, zip(range(4), CLASS_COLORS)):
        mask = labels == k
        x = df.loc[mask, "REDSHIFT"].dropna()
        y = df.loc[mask, "FRACDEV_R"].reindex(x.index)
        
        hexbin = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.1)
        ax.set_title(CLASS_NAMES[k], fontsize=9)
        ax.set_xlabel("Redshift")
        if k == 0:
            ax.set_ylabel("FRACDEV_R")
            
        fig.colorbar(hexbin, ax=ax, label="count")
        
    fig.suptitle("Redshift vs. FRACDEV_R by Class", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, "output/preprocessing/plots/redshift_fracdev_scatter.png", save)
    
    
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
    out["gz2_class_name"] = labels.map(CLASS_NAMES).fillna("excluded")
    
    retained = out[out["gz2_class"] >= 0].reset_index(drop=True)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    retained.to_csv(out_path, index=False)
    
    print(f"\nSaved labeled CSV: {out_path}  ({len(retained):,} rows)")
    print(f"Columns: {list(retained.columns)}")
    
    
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
        
    hart_path = argv[1]
    samples_path = argv[2]
    mapping_path = argv[3]
    out_path = argv[4] if len(argv) > 4 and not argv[4].startswith("--") else "data/gz2/processed/gz2_labeled.csv"
    
    save_figs = "--save" in argv
    
    # Build joined master table
    df = build_master(hart_path, samples_path, mapping_path)
    
    # Audit images 
    df = audit_images(df, "data/gz2/images")
    
    # Get and print summaries
    print_summarize_vote_fractions(df)
    print_threshold_sweep(df)
    
    # Assign labels at defaults to start
    print(f"\nAssigning labels at thresholds: {THRESHOLDS}")
    labels = assign_labels(df, THRESHOLDS)
    print_class_balance_report(labels, title="Class balance at defaults thresholds")
    
    # Re-sync labels after filtering - forgot to do this and it really threw me off
    df = df[labels >= 0].reset_index(drop=True)
    labels = assign_labels(df, THRESHOLDS)      # Done twice so that dropped rows are reported
    
    # Print covariate analysis on final labeled set
    print_covariate_summary(df, labels)
    print_domain_split_sweep(df, labels)
    
    # Generate plots
    if save_figs:
        print("\nSaving plots to output/preprocessing/plots.")
        print("Generating plots...")
    else: 
        print("\nGenerating plots...")
    plot_vote_distributions(df, THRESHOLDS, save=save_figs)
    plot_class_balance(labels, save=save_figs)
    plot_threshold_sweep(df, save_figs)
    plot_redshift_distribution(df, labels, save=save_figs)
    plot_fracdev_distribution(df, labels, save=save_figs)
    plot_redshift_fracdev_scatter(df, labels, save=save_figs)
    
    
    # Export new CSV
    export_labeled_csv(df, labels, out_path)
    
    
if __name__ == "__main__":
    main(sys.argv)
