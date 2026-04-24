"""
Danyal Ahmed - April 2026

train.py
Training loop for GZ2-based models. Supports two training regimes via the --loss flag:
    ce : hard-label cross-entropy against gz2_class
    kl : soft-label KL divergence against branch-product soft targets, 
         emulating GZ2 decision tree vote fractions
         
Each run writes a self-contained directory under output/runs/ named with its
hyperparameters. Running twice with identical hyperparameters will print a warning
and overwrite the previous outputs (pass --force to suppress this warning)

Usage:
    python train.py --loss ce --splits-dir data/gz2/processed/splits/soft
    python train.py --loss kl --splits-dir data/gz2/processed/splits/soft
    
    * Override defaults:
    python train.py --loss ce --epochs 30 --lr 1e-3 --batch-size 64
    
New loss types can be added by editing 'LOSS_REGISTRY' and the per-step loss
computation in `train_one_epoch`
"""

import sys
import json
import csv
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from constants import DEFAULT_SEED, CLASS_NAMES, N_CLASSES
from dataset import (
    GZ2Dataset,
    build_train_transform, build_eval_transform, build_weighted_sampler, build_ood_loader
)

from models import BasicCNN

#################### Loss Registry ####################

# Maps a loss name to the function that computes it per-batch
# Each entry must take (logits, hard_labels, soft_labels), and returns a scalar loss
# Extend the dict to add new loss types later

def _ce_loss(logits: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss against hard labels. Ignores soft targets."""
    return F.cross_entropy(logits, hard)

def _kl_loss(logits: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """KL divergence of model distribution from soft target distribution.
    
    I had to change this to manually compute, since F.kl_div has some strange NaN
    behavior. 
    
        KL(q || p) = -sum(q * log(p)) + sum(q * log(p))
        
    Since the second term is generally constant wrt the model, I drop it to instead minimize the 
    first term, which is essentially soft-target cross entropy and equivalent to minimizng KL
    
    Args:
        logits (torch.Tensor): raw input logits
        hard (torch.Tensor): IGNORED
        soft (torch.Tensor): soft-target vote fractions
        
    
    Note that F.kl_div expects log-probabilities as input and probabilities as target.
    batchmean averages over the batch dimension, which seems to match the per-sample
    interpretation I've seen in many papers
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft * log_probs).sum(dim=1).mean()

LOSS_REGISTRY = {
    "ce": _ce_loss,
    "kl": _kl_loss
}


#################### DATA HANDLING ####################

####                Config Dataclass                ####

@dataclass
class RunConfig:
    """All hyperparameters and run metadata for one training run
    
    This gets serialized to a config.json in the run directory so that every
    run's configuration is self-contained.
    
    I'm a big fan of this pattern in general, especially when trying to
    reproduce results later
    """
    loss: str
    splits_dir: str
    image_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    num_workers: int
    dropout: float
    eta_min: float
    run_name: str
    run_dir: str
    

def make_run_name(args) -> str:
    """Build a directory name for the run based on its hyperparameters
    
    Encodes the hyperparameters that should vary from run-to-run
    eg 'ce_lr3e-4_bs128_e30'
    
    Not the prettiest but I didn't feel like forcing a scan of the directory
    structure for unique names and auto-iteration
    """
    # Format LR: 3e-4 -> "3e-4", 0.001 -> "1e-3"
    lr_str = f"{args.lr:.0e}".replace('e-0', "e-")
    return f"{Path(args.splits_dir).stem}_{args.loss}_lr{lr_str}_bs{args.batch_size}_e{args.epochs}"


####                Data Loading                ####

def build_loaders(
    splits_dir: Path, image_dir: Path,
    batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Construct train/val/test DataLoaders from splits directory
    
    Splits directory must have train.csv, val.csv, test.csv, and stats.json
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader, dict]: 
            train_loader, val_loader, test_loader, statistics (from stats.json)
    """
    stats = json.loads((splits_dir / "stats.json").read_text())
    
    mean = stats["normalization"]["mean"]
    std = stats["normalization"]["std"]
    
    train_tf = build_train_transform(mean, std)
    eval_tf = build_eval_transform(mean, std)       # Basically just no shuffle or augmentation
    
    train_dataset = GZ2Dataset(splits_dir / "train.csv", image_dir, transform=train_tf)
    val_dataset = GZ2Dataset(splits_dir / "val.csv", image_dir, transform=eval_tf)
    test_dataset = GZ2Dataset(splits_dir / "test.csv", image_dir, transform=eval_tf)
    
    sampler = build_weighted_sampler(train_ds=train_dataset)
    
    # I'm training on Windows, which means that workers spawn() instead of fork(),
    # which takes a while. Setting persistence to True (when using multiple workers) 
    # means they persist between epochs
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        pin_memory=True     # page-locks the batch's tensors (faster transfer to GPU)
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=max(num_workers // 2, 1), persistent_workers=(num_workers > 0),
        pin_memory=True     
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=max(num_workers // 2, 1), persistent_workers=(num_workers > 0),
        pin_memory=True     
    )
    
    return train_loader, val_loader, test_loader, stats


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"


#################### TRAINING FUNCTIONS ####################

def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn , device: torch.device
    ) -> dict[str, float]:
    """Run a single training epoch. Returns metrics dictionary

    NOTE: 
    loss_fn should be one of the functions in LOSS_REGISTRY at the top, with signature:
        `func(logits: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor`

    Returns:
        {"loss": float, "accuracy": float}
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(loader, leave=False)
    
    for batch in progress_bar:
        imgs, hard, soft, _ = batch
        
        imgs=imgs.to(device, non_blocking=True)     # non_blocking ONLY WORKS WITH PINNED MEMORY
        hard=hard.to(device, non_blocking=True)     
        soft=soft.to(device, non_blocking=True)     
        
        logits = model(imgs)
        loss = loss_fn(logits, hard, soft)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Debugging due to issues with KL loss resolving to NaN
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")
        
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == hard).sum().item()
            total_samples += hard.size(0)
            total_loss += loss.item() * hard.size(0)
    
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }
    
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> dict[str, float]:
    """Run evaluation. Returns metrics object.

    Use same loss function as training so that the curves are directly comparable.
    
    loss_fn should be one of the functions in LOSS_REGISTRY at the top, with signature:
        `func(logits: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor`

    Returns:
        dict[str, float]: _description_
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    per_class_correct = np.zeros(N_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(N_CLASSES, dtype=np.int64)
    
    for batch in loader:
        imgs, hard, soft, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)
        
        logits = model(imgs)
        loss = loss_fn(logits, hard, soft)
        
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == hard).sum().item()     # Comparison is done against hard labels
        total_samples += hard.size(0)
        total_loss += loss.item() * hard.size(0)
        
        # per class accuracy
        for k in range(N_CLASSES):
            mask = (hard == k)
            per_class_total[k] += mask.sum().item()
            per_class_correct[k] += (predictions[mask]==k).sum().item()
            
    per_class_acc = {
        f"acc_{CLASS_NAMES[k].replace(' ' , '_').lower()}"
        :
            float(per_class_correct[k] / per_class_total[k]) if per_class_total[k] > 0 else 0.0
        for k in range(N_CLASSES)
    }
    
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        **per_class_acc     # Using pointers in Python feels weird sometimes
    }
    
    
class CSVLogger:
    """Per-epoch CSV writer. Fieldnames are fixed on first write"""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        
    def log(self, row: dict) -> None:
        # If file has not been created yet, set fieldnames and write the header
        if self._file is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()      # I don't think this should slow down training too much, 
                                # may remove if I need to tune performance
             
                                
    def close(self) -> None:
        """Close file (allows reuse)"""
        if self._file is not None:
            self._file.close()
            self._file = None
                

def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, metrics: dict
) -> None:
    """Saves a training checkpoint, so you can resume training if needed
    
    Useful for finding the best training accuracy over time (if overfit or gradients explode)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "metrics": metrics
    }, path)
    
    
####                Plotting                ####

def plot_curves(log_csv: Path, out_dir: Path, run_name: str) -> None:
    """Read the per-epoch CSV and generate a summary plot

    Produces three plots in out_dir/plots/:
        loss_curves.png     - training vs. validation loss over epochs
        accuracy_curves.png - training vs. validation accuracy over epochs
        lr_schedule.png     - learning rate over epochs
    """
    import pandas as pd
    df = pd.read_csv(log_csv)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss Curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train", color="steelblue")
    ax.plot(df["epoch"], df["val_loss"], label="val", color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss curves: {run_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curves.png", dpi=200)
    plt.close(fig)
    
    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["train_accuracy"], label="train", color="steelblue")
    ax.plot(df["epoch"], df["val_accuracy"],   label="val",   color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(f"Accuracy curves — {run_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_curves.png", dpi=150)
    plt.close(fig)
 
    # Learning rate schedule
    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["epoch"], df["lr"], color="darkgreen")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"LR schedule — {run_name}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "lr_schedule.png", dpi=150)
        plt.close(fig)
 
    print(f"    Wrote plots to {plots_dir}")
    
    
#################### MAIN ####################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--loss", required=True, choices=list(LOSS_REGISTRY.keys()),
                        help="Loss type: 'ce' (hard labels) or 'kl' (soft labels)")
    parser.add_argument("--splits-dir", required=True, type=str,
                        help="Diectory with train/val/test.csv and stats.json")
    
    # Paths
    parser.add_argument("--image-dir", type=str, default="data/gz2/images")
    parser.add_argument("--output-root", type=str, default="output/runs/",
                        help="Parent directory under which each run should get its own folder")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--eta-min", type=float, default=0.0,
                        help="Minimum LR for cosine annealing")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    
    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite of existing run directory without warning")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    run_name = make_run_name(args)
    run_dir = Path(args.output_root) / run_name
    
    # Check if directory exists, and if the user wants to overwrite it
    if run_dir.exists() and not args.force:
        print(f"WARNING: {run_dir} already exists. Overwrite? (use --force to suppress this warning)")
        while True:
            answer = input("[y/n] ")
            if answer.lower() in ['y', 'yes']:
                break
            elif answer.lower() in ['n', 'no']:
                print("Exiting...")
                sys.exit()
            else:
                print("Unknown input")
                
    run_dir.mkdir(parents=True, exist_ok=True)
            
    cfg = RunConfig(
        loss = args.loss, splits_dir=args.splits_dir, image_dir=args.image_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        num_workers=args.num_workers, dropout=args.dropout,
        eta_min=args.eta_min, run_name=run_name, run_dir=str(run_dir)
    )
    
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))     # This line was annoying to debug
    
    print(f"\nRun directory: {run_dir}")
    print(f"Loss: {args.loss}       Device: {args.device}")
    print(f"Epochs: {args.epochs}   Batch size: {args.batch_size}")
    print(f"LR: {args.lr}           Weight decay: {args.weight_decay}")
    
    # Get data
    print(f"\nBuilding DataLoaders from {args.splits_dir}...")
    train_loader, val_loader, test_loader, stats = build_loaders(
        Path(args.splits_dir), Path(args.image_dir), batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Check that soft targets were not errored out into one-hot vectors (see prepare_splits.py and dataset.py)
    if args.loss == "kl" and not train_loader.dataset.has_soft_targets:
        raise ValueError(
            "[--loss kl] required soft targets in the CSV. "
            "Re-run prepare_splits.py with --soft-targets or use [--loss ce] for hard-label training"
        )
        
    print(f"    train: {len(train_loader.dataset):,}  val: {len(val_loader.dataset):,}  test: {len(test_loader.dataset):,}")
    
    model = BasicCNN(num_classes=N_CLASSES, dropout=args.dropout).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BasicCNN   ({n_params:,} parameters)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    loss_func = LOSS_REGISTRY[args.loss]
    
    logger = CSVLogger(run_dir / "train_log.csv")
    
    ####        TRAINING LOOP       ####
    best_val_loss = float("inf")
    best_epoch = -1
    print(f"\nStarting training for {args.epochs} epochs...\n")
    t_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_func, args.device)
        val_metrics = evaluate(model, val_loader, loss_func, args.device)
        
        scheduler.step()
        
        # Logging
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            **{k: v for k, v in val_metrics.items() if k.startswith("acc_")},
            "epoch_seconds": time.time() - t_epoch
        }
        
        logger.log(row)
        
        per_class = " ".join(
            f"{CLASS_NAMES[k][:4]}={val_metrics[f'acc_' + CLASS_NAMES[k].replace(' ', '_').lower()]:.2f}"
            for k in range(N_CLASSES)
        )
        
        # Print stats
        print(f"Epoch {epoch:3d} / {args.epochs}  lr={current_lr:.2e}  train_loss={train_metrics["loss"]:.4f}  "
              f"val_loss={val_metrics["loss"]:.4f}  val_acc={val_metrics['accuracy']:.4f}  [{per_class}]  "       
              f"({row['epoch_seconds']:.1f}s)"
        )
        
        # Save and update best validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, scheduler, epoch, val_metrics)
            
    # Save final state
    save_checkpoint(run_dir / "checkpoint_last.pt", model, optimizer, scheduler, args.epochs, val_metrics)
    logger.close()
    
    
    t_total = time.time() - t_start
    print(f"\nTraining complete in {t_total/60:.1f} minutes.")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    print(f"\nGenerating plots...")
    plot_curves(run_dir / "train_log.csv", run_dir, run_name)
    
    # FInal summary
    print(f"\n--- Summary ---")
    print(f"Run directory: {run_dir}")
    print(f"Best validation loss: {best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"Final validation accuracy: {val_metrics["accuracy"]:.4f}")
    print(f"All artifacts written to {run_dir}/")
    
    
if __name__ == "__main__":
    main()