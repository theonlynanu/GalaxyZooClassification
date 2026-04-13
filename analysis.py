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
import matplotlib.gridspec as gridspec

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
    analysis
    

    Args:
        path (Path): csv filepath

    Returns:
        pd.DataFrame: Pandas DataFrame containing all rows with relevant columns
    """
    print(f"Loading samples metadata: {path}")
    cols = [SAMPLES_KEY, "REDSHIFT", "PETROR50_R", "PETROMAG_R", "PETROMAG_MR", "REGION"]
    
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


def audit_images(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Cross-reference labeled DataFrame against available images

    Args:
        df (pd.DataFrame): labeled and joined DataFrame
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
    
    missing = (~exists).sum()   # Someone said they're unfamiliar with this syntax --- '~' is bitwise NOT
    print(f"    {exists.sum():,} images found, {missing:,} missing ({100*missing / len(df):.1f}%)")
    
    if missing > 0:
        print("    Missing counts per-class:")
        for k, name in CLASS_NAMES.items():
            n_missing = ((~exists) & (df["gz2_class"] == k)).sum()
            n_total = (df["gz2_class"] == k).sum()
            print(f"        {name:<22} {n_missing:>6,} missing / {n_total:>6,} total ({100*n_missing/n_total:.1f}%)")
    
    return df[exists].reset_index(drop=True)


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


####        Label Construction          ####

def assign_labels(df: pd.DataFrame, threshold: dict[str: float]) -> pd.Series:
    """Assigns 4 class labels using the GZ2 decision tree paths
    
    0   Elliptical              T01=smooth
    1   Edge-on disk            T01=featured    AND     T02=edge-on
    2   Face-on spiral          T01=featured    AND     T02=not-edge-on     AND     T04=spiral
    3   Face-on non-spiral      T01=featured    AND     T02=not-edge-on     AND     T04=no-spiral
    -1  Ambiguous
    
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
    
    # Get and print summaries
    print_summarize_vote_fractions(df)
    print_threshold_sweep(df)
    
    # Assign labels at defaults to start
    print(f"\nAssigning labels at thresholds: {THRESHOLDS}")
    labels = assign_labels(df, THRESHOLDS)
    print_class_balance_report(labels, title="Class balance at defaults thresholds")
    
    # Filter to only rows that have associated images
    df["gz2_class"] = labels
    df=audit_images(df[df["gz2_class"] >= 0].reset_index(drop=True), "data/gz2/images")
    
    # Re-sync labels after filtering - forgot to do this and it really threw me off
    labels = df["gz2_class"]
    
    # Generate plots
    if save_figs:
        print("\nSaving plots to output/preprocessing/plots.")
    print("Generating plots..." if save_figs else "\nGenerating plots...")
    plot_vote_distributions(df, THRESHOLDS, save=save_figs)
    plot_class_balance(labels, save=save_figs)
    plot_threshold_sweep(df, save_figs)
    plot_redshift_distribution(df, labels, save=save_figs)
    
    
    # Export new CSV
    export_labeled_csv(df, labels, out_path)
    
    
if __name__ == "__main__":
    main(sys.argv)
