"""
Danyal Ahmed - April 2026
 
dataset.py
PyTorch Dataset class for GalaxyZoo2 image classification.
 
Reads a split CSV produced by prepare_splits.py and returns, per item:
    (image_tensor, hard_label_int, soft_label_tensor)
 
The caller needs to self configure the transform pipeline and pass it in, including
any variations for validation, testing, or training.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T
from PIL import Image

from constants import N_CLASSES, CLASS_NAMES


################### CONSTANTS ###################

# Soft target column names in split CSVs (needs to match prepare_splits.py output)
SOFT_COLS = [f"soft_{k}" for k in range(N_CLASSES)]


################### BUILDERS AND CLASSES ###################

####                Transformation Builders             ####

def build_train_transform(
    mean: list[float], std: list[float], crop_size: int = 224
) -> T.Compose:
    """Build a training transform pipeline
    
    rotate -> flip -> flip -> crop -> normalize
    
    Rotation happens on full 424x424 image so corner artifacts get trimmed
    by the center crop

    Args:
        mean (list[float]): per-channel RGB mean scaled to [0,1]
        std (list[float]): per-channel RGB stdev scaled to [0,1]
        crop_size (int, optional): center-crop size in pixels. Defaults to 224.

    Returns:
        T.Compose: a torchvision v2 Compose pipeline that takes a PIL image
        and returns a normalized float tensor of shape (3, crop_size, crop_size)
        (depth-first)
    """
    return T.Compose([
        T.ToImage(),    # PIL -> tensor
        T.RandomRotation(degrees=180, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.CenterCrop(crop_size),
        T.ToDtype(torch.float32, scale=True),   # uint8 [0,255] -> float32 [0,1]
        T.Normalize(mean=mean, std=std)
    ])
    
    
def build_eval_transform(
    mean: list[float], std: list[float], crop_size: int = 224
) -> T.Compose:
    """Build an evaluation transform pipeline
    
    Does not perform any augmentation

    Args:
        mean (list[float]): per-channel RGB mean scaled to [0,1]
        std (list[float]): per-channel RGB stdev scaled to [0,1]
        crop_size (int, optional): center-crop size in pixels. Defaults to 224.

    Returns:
        T.Compose: a torchvision v2 Compose pipeline
    """
    return T.Compose([
        T.ToImage(),
        T.CenterCrop(crop_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std)
    ])
    

####                Dataset Class               ####

class GZ2Dataset(Dataset):
    """PyTorch Dataset for GalaxyZoo2 image classification

    Each item yields a 3 tuples (image, hard_label, soft_label):
        image: normalized float32 tensor of shape (3, H, W)
        hard_label: int64 scalar tensor, each a class in [0, N_CLASSES)
        soft_label: float32 tensor of shape (N_CLASSES,), sums to 1.0
        
    Note that if the CSV does not have any soft_k columns (prepare_splits.py run
    without --soft-targets), then a one-hot vector is derived in its place, and
    self.has_soft_targets is set to False.
    """
    
    def __init__(
        self, csv_path: str | Path, image_dir: str | Path, transform: T.Compose | None,
        image_suffix: str = ".jpg", meta_cols: list[str] | None = None
    ):
        """
        Args:
            csv_path (str | Path): path to data csv
            image_dir (str | Path): directory containing <asset_id>.jpg files
            transform (T.Compose | None): torchvision v2 transform pipeline.
                If None, returns the raw PIL image
            image_suffix (str, optional): image file extension. Defaults to ".jpg".
            meta_cols (list[str], optional): list of metadata column names to 
                include, must already be present in data csv
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_suffix = image_suffix
        self.meta_cols = meta_cols or []
        
        df = pd.read_csv(self.csv_path)
        
        self.meta = {c: df[c].to_numpy() for c in self.meta_cols if c in df.columns}
        
        # Find required columns
        if "asset_id" not in df.columns or "gz2_class" not in df.columns:
            raise ValueError(
                f"{csv_path} missing required asset_id and/or gz2_class columns"
            )
            
        # Extract arrays ahead of time to avoid repeated lookups
        self.asset_ids = df["asset_id"].astype(str).to_numpy()
        self.hard_labels = df["gz2_class"].to_numpy(dtype=np.int64)
        
        # Soft targets are optional
        self.has_soft_targets = all(c in df.columns for c in SOFT_COLS)
        if self.has_soft_targets:
            self.soft_labels = df[SOFT_COLS].to_numpy(dtype=np.float32)
            if np.isnan(self.soft_labels).any():
                n_bad = np.isnan(self.soft_labels).any(axis=1).sum()
                raise RuntimeError(
                    f"{n_bad} rows in {csv_path} have NaN soft targets. "
                    f"Regenerate with prepare_splits.py after fixing compute_soft_targets."
                )
        else:
            self.soft_labels = np.eye(N_CLASSES, dtype=np.float32)[self.hard_labels]
            
        # Sample weights are optional (only used for training)
        self.sample_weights = (
            df["sample_weight"].to_numpy(dtype=np.float64).copy() if "sample_weight" in df.columns else None
        )
        
    def __len__(self) -> int:
        return len(self.asset_ids)
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset_id = self.asset_ids[index]
        path = self.image_dir / f"{asset_id}{self.image_suffix}"
        
        # load image
        img = Image.open(path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        hard = torch.as_tensor(self.hard_labels[index], dtype=torch.long)
        soft = torch.as_tensor(self.soft_labels[index], dtype=torch.float32)
        meta = {c: self.meta[c][index] for c in self.meta_cols}
        
        return img, hard, soft, meta
    

####                Training Sampler Builder                ####

def build_weighted_sampler(train_ds: GZ2Dataset, num_samples: int | None = None) -> WeightedRandomSampler:
    """Constructs a WeightedRandomSampler from the training dataset's sample_weights

    Args:
        train_ds (GZ2Dataset): training dataset with populated sample_weights
        num_samples (int | None, optional): Number of samples drawn per epoch. 
            Defaults to len(train_ds), which gives one full epoch of weighted sample draws

    Returns:
        WeightedRandomSampler: sampler for DataLoader
    """
    if train_ds.sample_weights is None:
        raise ValueError(
            "GZ2Dataset has no sample_weights. "
            "Ensure this CSV was properly produced by prepare_splits.py for training"
        )
        
    if num_samples is None:
        num_samples = len(train_ds)
        
    weights = torch.as_tensor(train_ds.sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


####                Out-of-Domain DataLoader                ####

def build_ood_loader(hard_csv: Path, image_dir: Path, stats_path: Path, batch_size: int=64, num_workers:int = 4) -> DataLoader:
    """Build a DataLoader for out-of-domain data (hard.csv from analysis.py)
    
    Assumes that out-of-domain loader is for evaluation/testing, not training

    Args:
        hard_csv (Path): filepath for out-of-domain data
        image_dir (Path): directory containing <asset_id>.jpg images
        stats_path (Path): filepath for dataset statistics
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.

    Returns:
        DataLoader: DataLoader for out-of-domain testing/validation
    """
    stats = json.loads(Path(stats_path).read_text())
    mean, std = stats["normalization"]["mean"], stats["normalization"]["std"]
    tf = build_eval_transform(mean, std)
    ds = GZ2Dataset(hard_csv, image_dir, transform=tf, meta_cols=["REDSHIFT", "FRACDEV_R"])
    
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


################### TESTING AND DEBUGGING ###################

def _smoke_test(splits_dir: Path, image_dir: Path, show_batch: bool = False) -> None:
    """Verifies the Dataset loads cleanly and produces good batches
    
    Prints summary information, iterates a few batches, and can display a batch visually
    """
    
    print(f"Smoke test on splits in {splits_dir}")
    print(f"Image directory:    {image_dir}")
    
    # Load normalization stats
    stats_path = splits_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    mean = stats["normalization"]["mean"]
    std = stats["normalization"]["std"]
    print(f"Normalization mean: {mean}")
    print(f"Normalization std: {std}")
    
    train_tf = build_train_transform(mean, std)
    eval_tf = build_eval_transform(mean, std)
    
    splits = [
        ("train", splits_dir / "train.csv", train_tf, []),
        ("val", splits_dir / "val.csv", eval_tf, []),
        ("test", splits_dir / "test.csv", eval_tf, []),     # Eval and testing use the same transformation pipeline
        ("ood (hard)", splits_dir.parent / "easyhard" / "hard.csv", eval_tf, ["REDSHIFT", "FRACDEV_R"])
    ]
    
    for name, csv_path, transform, meta_cols in splits:
        print(f"\n[{name}] loading {csv_path}")
        ds = GZ2Dataset(csv_path=csv_path, image_dir=image_dir, transform=transform, meta_cols=meta_cols)
        print(f"    len = {len(ds):,}")
        print(f"    has_soft_targets = {ds.has_soft_targets}")
        print(f"    has_sample_weights = {ds.sample_weights is not None}")
        
        # pull an item and check shape/datatypes
        img, hard, soft, meta = ds[0]
        print(f"    [item 0] image shape = {tuple(img.shape)}  dtype={img.dtype}")
        print(f"             hard label = {hard.item()}  ({CLASS_NAMES.get(hard.item(), "?")})")
        print(f"             soft label = {soft.tolist()}  sum={soft.sum().item():.4f}")
        print(f"         image min/max/mean = "
              f"{img.min().item():.3f} / {img.max().item():.3f} / {img.mean().item():.3f}")
        if "REDSHIFT" in meta:
            print(f"    Redshift = {meta["REDSHIFT"]}")
        if "FRACDEV_R" in meta:
            print(f"    Fracdev (r-band) = {meta["FRACDEV_R"]}")
        
        # build DataLoader and iterate a few times
        if name == "train":
            sampler = build_weighted_sampler(ds)
            loader = DataLoader(ds, batch_size=32, sampler=sampler, num_workers=2)
        else:
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            
        print(f"    iterating 2 batches from DataLoader...")
        for i, (imgs, hards, softs, metas) in enumerate(loader):
            print(f"    batch{i}: imgs={tuple(imgs.shape)} "
                  f"hards={tuple(hards.shape)} softs={tuple(softs.shape)} "
                  f"class_counts={np.bincount(hards.numpy(), minlength=N_CLASSES).tolist()}"
            )
            if i >= 1:
                break
            
        if show_batch and name == "train":
            _display_batch(imgs, hards, mean, std)
            
    print("\nSmoke test complete.")
    
    
def _display_batch(imgs: torch.Tensor, hards: torch.Tensor, mean: list[float], std: list[float]) -> None:
    """Display a grid of images from a batch, undoing normalization"""
    import matplotlib.pyplot as plt
    
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    
    n = min(16, imgs.shape[0])
    ncols = 4
    nrows = (n + ncols -1 ) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.flatten()
    
    for i in range(n):
        # Undo normalization
        img = imgs[i] * std_t + mean_t
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        
        # Thought I was having trouble with visibility initially
        # img = np.clip(img * 1, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"class {hards[i].item()}", fontsize=8)
        axes[i].axis("off")
        
    for i in range(n, len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--splits-dir", type=str, 
                        required=True, help="directory containing train/val/test.csv and stats.json"
    )
    parser.add_argument("--image-dir", type=str,
                        default="data/gz2/images",
                        help="directory containing <asset_id>.jpg files"
    )
    parser.add_argument("--show-batch", action="store_true",
                        help="display a sample training batch as an image grid"
    )
    args = parser.parse_args()
    
    _smoke_test(Path(args.splits_dir), Path(args.image_dir), show_batch=args.show_batch)