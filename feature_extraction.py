"""
Danyal Ahmed - April 2026

feature_extraction.py
Extracts a fixed-length morphology feature vector from each GZ2 image
to be used by classical ML models (logistic regression, SVM, etc)

Feature vector composition (207 features):
    [0]         Concentration index (C)
    [1]         Asymmetry (A)
    [2]         Smoothness (S)
    [3]         Gini coefficient (G)
    [4]         M20
    [5]         Ellipticity
    [6]         Position angle (sin(2*theta) in [-1,1])
    [7]         Petrosian radius estimate
    [8:18]      Radial profile bins (10 bins)
    [18:207]    Spatial pyramid gradient histogram (3 levels of 9 bins)
                    9, 36, and 144 values per level, 189 total
                    
Computed on center-cropped float32 [0,1] image without per-channel normalization

Outputs one .npz file per split:
    X:      (n_samples, 207)        [float32]       feature matrix
    objids: (n_samples,)            [int64]         dr7objid for alignment checks
    labels: (n_samples,)            [int64]         gz2_class hard labels
"""

import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


#################### DEFAULTS/CONSTANTS ####################

DEFAULT_SPLITS_DIR = Path("data/gz2/processed/splits/domain")
DEFAULT_OOD_CSV = Path("data/gz2/processed/splits/easyhard/hard.csv")
DEFAULT_IMAGE_DIR = Path("data/gz2/images")
DEFAULT_CROP_SIZE = 224
DEFAULT_PROFILE_BINS = 10
DEFAULT_GRAD_BINS = 9
DEFAULT_WORKERS = 4

SMOOTHING_SIGMA = 0.25      # Smoothness kernel width as fraction of Petrosian radius
NOISE_PERCENTILE = 10       # Percentile used for background noise estimation
MIN_FLUX_THRESHOLD = 1e-6


#################### FEATURE CALCULATIONS ####################

def _center_crop(img: np.ndarray, crop_size:int) -> np.ndarray:
    """Center crop an HxWxC image to crop_size x crop_size x channels"""
    
    h, w = img.shape[:2]
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    
    return img[y0:y0 + crop_size, x0:x0 + crop_size]

def _luminance(img:np.ndarray) -> np.ndarray:
    """Convert float32 RGB image to luminance"""
    return img.mean(axis=2)

def _radial_flux(lum:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative flux as a function of radius from image center
    
    Essentially the curve of total luminosity as radius from the center
    increases. The rate of increase may indicate overall shape and density

    Args:
        lum (np.ndarray): float32 (H, W) luminance

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            sorted unique integer radii in pixels
            cumulative flux fraction at each radius
    """
    h, w = lum.shape
    cy, cx = h / 2.0, w / 2.0
    
    ys, xs = np.mgrid[0:h, 0:w]
    r_map = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(np.int32)
    
    total = lum.sum()
    if total < MIN_FLUX_THRESHOLD:
        return np.array([0]), np.array([1.0])
    
    max_r = r_map.max()
    cumulative = np.zeros(max_r + 1, dtype=np.float64)
    
    for r in range(max_r + 1):
        cumulative[r] = lum[r_map <= r].sum()
    cumulative /= total
    
    return np.arange(max_r + 1), cumulative
    

def _petrosian_radius(lum: np.ndarray, eta: float=0.2) -> float:
    """Estimate Petrosian radius
    
    Petrosian radius is the radius at which the ratio of local brightness in a
    ring at r to the mean surface brightness of everything within r reaches eta
    
    Essentially, "how far from the center until we have x% of the total luminosity?"
    (slight simplification but this is the basic idea)

    Args:
        lum (np.ndarray): (H, W) luminance image
        eta (float, optional): Ratio threshold. Defaults to 0.2.

    Returns:
        float: petrosian radius in pixels, or half the image size if none was found
    """
    h, w = lum.shape
    cy, cx = h / 2.0, w / 2.0
    max_r = int(min(h, w) / 2)
    
    ys, xs = np.mgrid[0:h, 0:w]
    r_map = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    
    total = lum.sum()
    if total < MIN_FLUX_THRESHOLD:
        return max_r / 2.0
    
    # For each radius, get two values:
    # - mean pixel value in ring between r-0.5 and r+0.5 pixels
    # - mean surface brightness within r
    for r in range(2, max_r):
        ring_mask = (r_map >= r - 0.5) & (r_map < r + 0.5)
        disk_mask = r_map < r
        
        # Get area of ring and total disk
        ring_area = ring_mask.sum()
        disk_area = disk_mask.sum()
        
        # Avoid divide by zero
        if ring_area == 0 or disk_area == 0:
            continue
        
        # Get surface brightness within ring and total circle
        local_sb = lum[ring_mask].sum() / ring_area
        mean_sb = lum[disk_mask].sum() / disk_area
        
        if mean_sb < MIN_FLUX_THRESHOLD:
            continue
        
        # If we reach eta, return
        if local_sb / mean_sb <= eta:
            return float(r)
        
    # If no valid radius is found, return half the size of the image
    return float(max_r / 2.0)


def compute_concentration(lum: np.ndarray) -> float:
    """Compute concentration
    
    C = 5 * log_10(r80/r20)
    
    Essentially, a high central concentration may be an elliptical, while
    a low central concentration may be a spiral

    Args:
        lum (np.ndarray): float32 (H, W) luminance

    Returns:
        float: concentration index
    """
    
    radii, cumulative = _radial_flux(lum)
    
    # Use search to find radii for 20% and 80% of total luminance
    r20_idx = np.searchsorted(cumulative, 0.20)
    r80_idx = np.searchsorted(cumulative, 0.80)
    
    # Convert to actual radius 
    r20 = float(radii[min(r20_idx, len(radii) - 1)])
    r80 = float(radii[min(r80_idx, len(radii) - 1)])
    
    if r20 < 1.0:
        r20 = 1.0
        
    return 5.0 * np.log10(r80 / r20)


def compute_asymmetry(lum: np.ndarray) -> float:
    """A = sum|I - I180| / (2 * sum|I|) - background correction
     
    Essentially, flip the image 180 degrees to determine difference of each pixel
    from its mirror position

    Args:
        lum (np.ndarray): float32 (H, W) luminance

    Returns:
        float: asymmetry index (clipped to range [0,1])
    """
    h, w = lum.shape
    rot = np.rot90(lum, 2)
    
    total = lum.sum()
    if total < MIN_FLUX_THRESHOLD:
        return 0.0
    
    A_raw = np.abs(lum - rot).sum() / (2.0 * total)
    
    # Correct for background noise using info from the corner
    corner_size = max(h // 8, 4)
    bg = lum[:corner_size, :corner_size]
    bg_rot = np.rot90(bg, 2)
    bg_total = bg.sum()
    
    if bg_total > MIN_FLUX_THRESHOLD:
        A_bg = np.abs(bg - bg_rot).sum() / (2.0 * total)
    else:
        A_bg = 0.0
        
    # Use corner regions as floor
    return float(np.clip(A_raw - A_bg, 0.0, 1.0))


def compute_smoothness(lum:np.ndarray, petrosian_r: float) -> float:
    """S = sum|I - I_smooth| / sum|I|
    
    The difference between the original luminance and the luminance smoothed
    by a Gaussian, normalized to [0,1]
    
    Low S indicates smooth distribution, high S indicates some sort of clumpy
    substructure (can't think of the word right now)
    """
    total = lum.sum()
    if total < MIN_FLUX_THRESHOLD:
        return 0.0
    
    # Size of Gaussian kernel is proportional to petrosian radius
    kernel_size = max(int(SMOOTHING_SIGMA * petrosian_r), 1)
    
    # Make sure kernel is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    smoothed = cv2.GaussianBlur(lum, (kernel_size, kernel_size), 0)
    S = np.abs(lum - smoothed).sum() / total
    
    return float(np.clip(S, 0.0, 1.0))


def compute_gini(lum: np.ndarray) -> float:
    """Computes the Gini coefficient of pixel flux distribution

    G = (2 * sum(i * f_i)) / (n * sum(f_i)) - (n + 1)/n

    This comes from economics - my undergrad was in econ and astrophysics, so this
    is extremely familiar
    
    Gini coefficient measures how far away from perfect equality a distribution 
    is. We apply that idea to the pixel flux by using the weighted sum of each 
    pixel's value times its rank over total flux (times pixel count). This is
    basically a Lorenz curve!
    
    The Gini coefficient is just the difference between this curve and a 45-degree
    line (perfect equality).

    Args:
        lum (np.ndarray): float32 (H x W) luminance

    Returns:
        float: Gini coefficient
    """
    flat = lum.flatten().astype(np.float64)
    flat = flat[flat > 0]       # filters out background pixels
    
    if len(flat) < 2:
        return 0.0
    
    flat = np.sort(flat)
    n = len(flat)
    idx = np.arange(1, n + 1)
    
    G = (2.0 * (idx * flat).sum()) / (n * flat.sum()) - (n + 1.0)/n
    
    return float(np.clip(G, 0.0, 1.0))


def compute_m20(lum: np.ndarray) -> float:
    """Compute second-order moment of the brightest 20% of pixels, normalized by
    total second-order moment
    
    M20 = log10(M_bright20 / M_total)
    
    Low would indicate bright pixels concentrated near center, high indicates
    bright off-center structures

    Args:
        lum (np.ndarray): float32 (H, W) luminance

    Returns:
        float: M20 value
    """
    
    h, w = lum.shape
    cy, cx = h / 2.0, w / 2.0
    
    ys, xs = np.mgrid[0:h, 0:w]
    r2_map = (ys - cy) ** 2 + (xs - cx) ** 2
    
    total = lum.sum()
    if total < MIN_FLUX_THRESHOLD:
        return -1.0
    
    M_total = (lum * r2_map).sum()
    if M_total < MIN_FLUX_THRESHOLD:
        return -1.0
    
    # Get brightest 20% of pixels
    threshold = np.percentile(lum, 80)
    bright = lum >= threshold
    
    M20_num = (lum[bright] * r2_map[bright]).sum()
    
    epsilon = 1e-10     # Numpy throws errors for division by very small floats, this is a sentinel
    val = np.log10(max(M20_num, epsilon) / max(M_total, epsilon)) if M20_num > 0 and M_total > 0 else -1.0
        
    return float(np.clip(val, -4.0, 0.0))


def compute_shape_moments(lum: np.ndarray) -> tuple[float, float]:
    """Compute ellipticity and position angle from intensity-weighted second moments

    Ellipticity E in [0,1] (0=circular, 1=infinitely elongated)
    Position angle is sin(2*theta) in [-1, 1] 
    
    I don't imagine angle is of particular interest to classification, but I am 
    curious to see if this affects performance

    Args:
        lum (np.ndarray): float32 (H, W) luminance

    Returns:
        tuple[float, float]: (ellipticity, sin(2*theta))
    """
    
    h, w = lum.shape
    cy, cx = h / 2.0, w / 2.0
    
    ys, xs = np.mgrid[0:h, 0:w]
    dy = ys - cy
    dx = xs - cx
    
    total = lum.sum()
    if total < MIN_FLUX_THRESHOLD:
        return 0.0,0.0
    
    Mxx = (lum * dx * dx).sum() / total
    Myy = (lum * dy * dy).sum() / total
    Mxy = (lum * dx * dy).sum() / total
    
    # Eigenvalues of moment matrix
    trace = Mxx + Myy
    det = Mxx * Myy - Mxy ** 2
    disc = np.sqrt(max((trace / 2) ** 2 - det, 0.0))
    
    # Major and minor axis variance
    a2 = trace / 2 + disc
    b2 = trace / 2 - disc
    
    b2 = max(b2, 0.0)
    
    denominator = a2 + b2
    ellipticity = float((a2 - b2) / denominator)  if denominator > MIN_FLUX_THRESHOLD else 0.0
    
    theta = 0.5 * np.arctan2(2 * Mxy, Mxx-Myy)
    sin_2theta = float(np.sin(2*theta))
    
    return float(np.clip(ellipticity, 0.0, 1.0)), sin_2theta


def compute_radial_profile(lum:np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Mean flux in n_bins concentric rings, normalized to sum to 1
    
    Essential how brightness varies with distance from the center.

    Args:
        lum (np.ndarray): float32 (H, W) luminance
        n_bins (int, optional): number of rings. Defaults to 10.

    Returns:
        np.ndarray: float32 (n_bins,) normalized profile vector
    """
    
    h, w = lum.shape
    cy, cx = h / 2.0, w / 2.0
    max_r = min(h, w) / 2.0
    
    ys, xs = np.mgrid[0:h, 0:w]
    r_map = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    
    # Get ring edges from 0 to max_radius
    edges = np.linspace(0, max_r, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    
    for i in range(n_bins):
        mask = (r_map >= edges[i]) & (r_map < edges[i + 1])
        if mask.sum() > 0:
            profile[i] = lum[mask].mean()
            
    total = profile.sum()
    if total > MIN_FLUX_THRESHOLD:
        profile /= total
        
    return profile


def compute_spatial_pyramid_gradients(lum: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """Spatial pyramid histogram of gradient orientations
    
    Three pyramid levels (1x1, 2x2, 4x4) with 21 total spatial regions.
    Each region contributes ahistogram of unsigned gradient orientations (0-180)
    weighted by gradient magnitude and L1 normalized.

    Args:
        lum (np.ndarray): float32 (H, W) luminance image
        n_bins (int, optional): number of orientation histograms per region. Defaults to 9.

    Returns:
        np.ndarray: concatenated pyramid descriptor
    """
    
    # Sobel gradients
    lum_u8 = (lum * 255).astype(np.uint8)
    Gx = cv2.Sobel(lum_u8, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(lum_u8, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    
    orientation = (np.degrees(np.arctan2(np.abs(Gy), Gx)) % 180).astype(np.float32)
    
    bin_edges = np.linspace(0, 180, n_bins + 1)
    h, w = lum.shape
    descriptor = []
    
    for n_cells in [1, 2, 4]:
        cell_h = h // n_cells
        cell_w = w // n_cells
        
        for row in range(n_cells):
            for col in range(n_cells):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w
                
                cell_mag = magnitude[y0: y1, x0:x1].flatten()
                cell_orientation = orientation[y0:y1, x0:x1].flatten()
                
                hist, _ = np.histogram(cell_orientation, bins=bin_edges, weights=cell_mag)
                
                # L1 normalization on each cell histogram
                hist_sum = hist.sum()
                if hist_sum > MIN_FLUX_THRESHOLD:
                    hist = hist / hist_sum
                    
                descriptor.append(hist.astype(np.float32))
                
    return np.concatenate(descriptor)



    
#################### FEATURE EXTRACTION ####################

def extract_features(
    asset_id: str,
    image_dir: Path,
    crop_size: int = DEFAULT_CROP_SIZE,
    n_profile_bins: int = DEFAULT_PROFILE_BINS,
    n_grad_bins: int = DEFAULT_GRAD_BINS,
    image_suffix: str = ".jpg"
) -> np.ndarray | None:
    """Extract the full morphology feature vector for a single galaxy image
    
    Returns None if the image cannot be loaded or is malformed
    
    Feature layout:
        [0]     concentration
        [1]     asymmetry
        [2]     smoothness
        [3]     Gini
        [4]     M20
        [5]     ellipticity
        [6]     sin(2 * position_angle)
        [7]     Petrosian radius
        [8:8*n] radial profile (n = n_profile_bins)
        [8+n:]  spatial pyramid gradient histogram
        
    Args:
        asset_id (str): image filename stem
        image_dir (Path): directory containing <asset_id>.jpg files
        crop_size (int): center crop size in pixels
        n_profile_bins (int): number of radial profile rings
        n_grad_bins (int): orientation bins in gradient histograms
        image_suffix (str): image file extension
        
    Returns:
        np.ndarray | None: float32 feature vector, or None on failure
    """
    path = Path(image_dir) / f"{asset_id}{image_suffix}"
    
    try:
        pil = Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError):
        return None
    
    img = np.asarray(pil, dtype=np.float32) / 255.0
    if img.ndim != 3 or img.shape[2] != 3:
        return None
    
    img = _center_crop(img, crop_size)
    lum = _luminance(img)
    
    # Get scalar features
    petrosian_r = _petrosian_radius(lum)
    concentration = compute_concentration(lum)
    asymmetry = compute_asymmetry(lum)
    smoothness = compute_smoothness(lum, petrosian_r)
    gini = compute_gini(lum)
    m20 = compute_m20(lum)
    ellipticity, sin_2theta = compute_shape_moments(lum)
    
    # Normalize Petrosian radius
    petrosian_norm = float(petrosian_r / (crop_size / 2.0))
    
    scalars = np.array([
        concentration, asymmetry, smoothness, gini, m20, ellipticity, sin_2theta, petrosian_norm
    ], dtype=np.float32)
    
    profile = compute_radial_profile(lum, n_profile_bins)
    gradient = compute_spatial_pyramid_gradients(lum, n_grad_bins)
    
    return np.concatenate([scalars, profile, gradient])


#################### WORKERS AND PROCESSING ####################

def _worker(row: tuple, image_dir: Path, crop_size: int, n_profile_bins: int, n_grad_bins: int) -> np.ndarray | None:
    """Worker for unpacking CSV rows and extracting features

    Args:
        row (tuple): (asset, objid) from row
        image_dir (Path): image directory
        crop_size (int): center crop size
        n_profile_bins (int): radial profile bins
        n_grad_bins (int): gradient orientation bins

    Returns:
        np.ndarray | None: feature vector or None on failure
    """
    asset_id, _ = row
    return extract_features(
        asset_id, image_dir, crop_size, n_profile_bins, n_grad_bins
    )
    
    
def process_split(
    csv_path: Path,
    image_dir: Path,
    out_path: Path,
    crop_size: int,
    n_profile_bins: int,
    n_grad_bins: int,
    n_workers: int,
    verify: bool = True
) -> None:
    """Extract features for all rows in a split CSV and save to .npz file

    Args:
        csv_path (Path): split CSV with asset_id, gz2_class, and dr7objid columns
        image_dir (Path): image directory
        out_path (Path): output .npz filepath
        crop_size (int): center crop size
        n_profile_bins (int): radial profile bines
        n_grad_bins (int): gradient orientation bins
        n_workers (int): parallel worker processes
        verify (bool, optional): whether to verify output after saving. Defaults to True.
    """
    
    print(f"\nProcessing {csv_path.name} -> {out_path.name}")
    
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"    {n:,} rows loaded")
    
    rows = list(zip(df["asset_id"].astype(str), df["dr7objid"]))
    worker_func = partial(
        _worker,
        image_dir=image_dir,
        crop_size=crop_size,
        n_profile_bins=n_profile_bins,
        n_grad_bins=n_grad_bins
    )
    
    results = []
    failed = 0
    
    with Pool(processes=n_workers) as pool:
        for vec in tqdm(pool.imap(worker_func, rows), total = n, desc=f"    {csv_path}", unit="img"):
            if vec is None:
                failed += 1
                results.append(np.full(8 + n_profile_bins + 21 * n_grad_bins, fill_value=np.nan, dtype=np.float32))
            else:
                results.append(vec)
                
                
    if failed:
        print(f"    WARNING: {failed:,} images failed - rows filled with NaN")
        
    X = np.stack(results, axis=0)
    objids = df["dr7objid"].to_numpy(dtype=np.int64)
    labels = df["gz2_class"].to_numpy(dtype=np.int64)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, objids=objids, labels=labels)
    print(f"    Saved {out_path}  shape={X.shape}  {X.nbytes / 1e6:.1f} MB")
    
    if verify:
        _verify(out_path, df)
        
        
def _verify(npz_path: Path, df: pd.DataFrame) -> None:
    """Load saved .npz file and verify row count and id alignment

    Args:
        npz_path (Path): path to .npz
        df (pd.DataFrame): original CSV DataFrame
    """
    
    data = np.load(npz_path)
    
    assert data["X"].shape[0] == len(df), f"Row count mismatch: {data['X'].shape[0]} vs {len(df)}"
    
    assert np.array_equal(data["objids"], df["dr7objid"].to_numpy(dtype=np.int64)), "objid misalignment: feature rows do not match CSV rows"
    
    assert not np.all(np.isnan(data["X"])), "Feature matrix is entire NaN"
    

    nan_rows = np.isnan(data["X"]).any(axis=1).sum()
    
    if nan_rows:
        print(f"    WARNING: {nan_rows:,} rows contain NaN values")
    else:
        print(f"    Verified: {data['X'].shape[0]:,} rows, {data['X'].shape[1]} features, no NaN")
        
        
#################### MAIN ####################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    
    # Path arguments
    parser.add_argument("--splits-dir", type=Path, required=True,
                        help=f"directory containing split CSVs")
    parser.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD_CSV,
                        help=f"path to OOD hard.csv (default: {DEFAULT_OOD_CSV})")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help=f"iamge directory (default: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"output directory for .npz files (default is same as --splits-dir)")
    
    # Behavioral arguments
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test", "ood"],
                        choices=["train", "val", "test", "ood"],
                        help="which splits to process (default: all)")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE,
                        help=f"center crop size in pixels (default: {DEFAULT_CROP_SIZE})")
    parser.add_argument("--profile-bins", type=int, default=DEFAULT_PROFILE_BINS,
                        help=f"number of radial profile rings (default: {DEFAULT_PROFILE_BINS})")
    parser.add_argument("--gradient-bins", type=int, default=DEFAULT_GRAD_BINS,
                        help=f"number of gradient orientation bins (default: {DEFAULT_GRAD_BINS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"number of parallel worker processes (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-verify", action="store_true", help="skip post-save validation")
    
    args = parser.parse_args()
    
    out_dir = args.out_dir if args.out_dir is not None else args.splits_dir
    
    # Report feature vector
    n_features = 8 + args.profile_bins + 21 * args.gradient_bins
    print(f"Feature extraction")
    print(f"    Splits to process   : {args.splits}")
    print(f"    Image directory     : {args.image_dir}")
    print(f"    Output directory    : {out_dir}")
    print(f"    Crop size           : {args.crop_size}px")
    print(f"    Feature vector      : {n_features} dimensions")
    print(f"        Scalar (CAS + Gini + M20 + shape + Petrosian)   : 8")
    print(f"        Radial profile bins                             : {args.profile_bins} bins")
    print(f"        Spatial Pyramid gradient                        : 21 * {args.gradient_bins} bins")
    print(f"    Workers             : {args.workers}")
    
    split_map = {
        "train" : args.splits_dir / "train.csv",
        "val" : args.splits_dir / "val.csv",
        "test" : args.splits_dir / "test.csv",
        "ood" : args.ood_csv
    }
    
    for split_name in args.splits:
        csv_path = split_map[split_name]
        out_path = out_dir / f"features_{split_name}.npz"
        
        if not csv_path.exists():
            print(f"\nSkipping {split_name}: {csv_path} not found ")
            continue
        
        process_split(
            csv_path=csv_path,
            image_dir=args.image_dir,
            out_path=out_path,
            crop_size=args.crop_size,
            n_profile_bins=args.profile_bins,
            n_grad_bins=args.gradient_bins,
            n_workers=args.workers,
            verify=not args.no_verify
        )
        
    print("\nFeature extraction complete.")
    
    
if __name__ == "__main__":
    main()