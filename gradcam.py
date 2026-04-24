"""
Danyal Ahmed - April 2026

gradcam.py

For visualizing gradients in trained models, largely for discerning the primary
regions of interest any number of models have learned.

Supports:
    - Single-model attention visualization for one or more images
    - Side-by-side comparison of 2+ models on the same images
    - Image selection by explicit asset_id, test-set index, or auto-picked 
      categories derived from model agreement/disagreement
      
Script is model-agnostic, only needing:
    - run directory (config.json + checkpoint_*.pt)
    - a model with a named target layer (default: block4 for BasicCNN)
    
Usage examples:
    # Single mode, single image by asset_id
        python gradcam.py --runs output/runs/soft_ce_lr3e-4_bs128_e30 \\
                          --splits-dir data/gz2/processed/splits/soft \\
                          --asset_ids 96684
                          
    # Compare two models on auto-selected boundary cases
        python gradcam.py --runs run_a run_b run_c \\
                          --splits-dir data/gz2/processed/splits/soft \\
                          --auto-select
                          --per-category 4
                          
    # Combining explicit and auto-selected images with a group name
        python gradcam.py --runs run_a run_b --splits-dir <splits_dir/> \\
                          --auto-select --group-name hand_picked
            This produces hand_picked.png 
                                     
"""
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from constants import CLASS_NAMES, N_CLASSES, DEFAULT_SEED
from dataset import GZ2Dataset, build_eval_transform
from models import BasicCNN
from evaluate import load_model_from_run, run_inference
from train import get_device


#################### CONFIG ####################

# Target layer attribute name within the model (accessible by model.[layer_name])
DEFAULT_TARGET_LAYER = "block4"

# Default number of examples to select per auto-category
DEFAULT_PER_CATEGORY = 4

# Overlay opacity for heatmap over display image
OVERLAY_ALPHA = 0.45

# Brightness multiplier for displayed galaxy images (reverse normalization makes images dark)
DISPLAY_BRIGHTNESS = 3.0


#################### STRUCTURES ####################

@dataclass
class RunArtifact:
    """Everything needed to visualize and describe a trained model.
    
        Attributes:
        name:      short identifier (e.g. run directory name)
        model:     loaded nn.Module in eval mode on the target device
        probs:     (N, num_classes) softmax probabilities on the test set
        preds:     (N,) argmax predictions
        loss_type: "ce" | "kl" | other (read from config.json, may be None)
    """
    name: str
    model: torch.nn.Module
    probs: np.ndarray
    preds: np.ndarray
    loss_type: str | None
    
    
class GradCAM:
    """Grad-CAM attention extractor for a given model and target layer.
    
    Hooks onto the target layer's forward pass (capture activations) and
    backward pass (capture gradients). Computes class-specific activation map
    as a spatially weighted sum of feature maps weighted by channel-wise mean gradient
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        
        # Forward hook for output activation, backward hook for output gradient
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)
        
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach().clone()
        
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()
        
    
    def compute(self, inputs: torch.Tensor, class_idx: int) -> np.ndarray:
        """Produce Grad-CAM heatmaps for the given input batch and class
        
        Args:
            inputs: (B, 3, H, W) normalized input tensor
            class_idx: integer class index to observe
            
        Returns:
            (B, H_feat, W_feat) numpy array w values in [0,1]
        """
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(inputs)
        
        logits[:, class_idx].sum().backward()
        
        weights = self.gradients.mean(dim=(2,3), keepdim=True)      # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)               # (B, H, W)
        cam = F.relu(cam)       # Keep only positive contributions
        
        cam_np = cam.cpu().numpy()
        
        for i in range(cam_np.shape[0]):
            c = cam_np[i]
            c_min, c_max = c.min(), c.max()
            cam_np[i] = (c - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(c)
            
        return cam_np
    
    
    def close(self) -> None:
        """Remove forward and backward hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()
        
        
#################### Image helpers ####################

def upsample_cam(cam_2d: np.ndarray, target_size: int) -> np.ndarray:
    """Bilinear upsample of a (H, W) heatmap to (target_size, target_size)"""
    t = torch.from_numpy(cam_2d).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    
    return up.squeeze().numpy()


def load_display_image(asset_id: str, image_dir: Path, crop_size: int = 224) -> np.ndarray:
    """Load an image at its center-cropped display size (unnormalize, [0,1])"""
    
    img = Image.open(image_dir / f"{asset_id}.jpg").convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    
    return arr[top: top + crop_size, left: left + crop_size]

def overlay_heatmap(
    base_rgb: np.ndarray, heatmap_2d: np.ndarray, alpha: float = OVERLAY_ALPHA
) -> np.ndarray:
    """Blend a [0,1] 2D heatmap onto a [0,1] RGB image"""
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap_2d)[...,:3]     # Drops alpha channel
    
    return (1 - alpha) * base_rgb + alpha * heat_rgb


def resolve_asset_ids_to_indices(asset_ids: list[str], all_asset_ids: np.ndarray) -> list[int]:
    """Converts requested asset_ids into test-set indices"""
    id_to_idx = {aid: i for i, aid in enumerate(all_asset_ids.astype(str))}
    missing = [aid for aid in asset_ids if aid not in id_to_idx]
    
    if missing:
        raise ValueError(
            f"Requested asset_ids not found in test set: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    return [id_to_idx[aid] for aid in asset_ids]


def validate_indices(indices: list[int], n_total: int) -> None:
    """Raise error if index is out of range"""
    bad = [i for i in indices if i < 0 or i >= n_total]
    if bad:
        raise ValueError(
            f"Indices out of range [0, {n_total}): {bad[:5]}"
            f"{'...' if len(bad) > 5 else ''}"
        )
        
        
def auto_select_categories(
    runs: list[RunArtifact], hard: np.ndarray, soft: np.ndarray, per_category: int,
    seed: int = DEFAULT_SEED
) -> dict[str, list[int]]:
    """Auto select interesting test-set indices for multi-model comparison.
    
    Categories:
        - all_agree_correct     : every model predicts the true class
        - all_wrong             : every model predicts the wrong class
        - models_disagree       : models do not all predict the same class
        - high_ambiguity        : max soft target < 0.5
    """
    rng = np.random.default_rng(seed)
    
    # Stack (n_runs, n) prediction matrices
    preds_matrix = np.stack([r.preds for r in runs])    # (R, N)
    correct_matrix = preds_matrix == hard
    all_correct = correct_matrix.all(axis=0)
    all_wrong = (~correct_matrix).all(axis=0)
    disagree = (preds_matrix != preds_matrix[0]).any(axis=0)
    high_ambiguity = soft.max(axis=1) < 0.5
    
    categories = {
        "all_agree_correct" : np.where(all_correct)[0],
        "all_wrong"         : np.where(all_wrong)[0],
        "models_disagree"   : np.where(disagree)[0],
        "high_ambiguity"    : np.where(high_ambiguity)[0]
    }
    
    selected: dict[str, list[int]] = {}
    
    for name, pool in categories.items():
        if len(pool) == 0:
            print(f"    Warning: no examples for category '{name}'")
            selected[name] = []
            continue
        
        take = min(per_category, len(pool))
        picked = rng.choice(pool, size=take, replace=False)
        selected[name] = sorted(picked.tolist())
        print(f"    {name}: {len(pool)} available, selected {take}")
        
    return selected


#################### Figures/Plots ####################

def plot_group(
    group_name: str, indices: list[int], runs: list[RunArtifact], hard: np.ndarray,
    soft: np.ndarray, asset_ids: np.ndarray, test_ds: GZ2Dataset, image_dir: Path,
    device: str, target_layer_name: str, output_path: Path
) -> None:

    if not indices:
        print(f"    Skipping {group_name} (no indices)")
        return
    
    n_images = len(indices)
    n_cols = 1 + len(runs)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(3.5 * n_cols, 3.2 * n_images), squeeze=False)
    
    # Build extractors (one per model)
    cams = [GradCAM(r.model, getattr(r.model, target_layer_name)) for r in runs]
    
    try:
        for row_idx, idx in enumerate(indices):
            img_tensor, hard_i, soft_i, _ = test_ds[idx]
            asset_id = str(asset_ids[idx])
            input_batch = img_tensor.unsqueeze(0).to(device)
            
            display = load_display_image(asset_id, image_dir)
            display_bright = np.clip(display * DISPLAY_BRIGHTNESS, 0, 1)
            
            true_cls_name = CLASS_NAMES[int(hard_i)]
            soft_str = " ".join(f"{s:.2f}" for s in soft_i.tolist())
            
            ax = axes[row_idx, 0]
            ax.imshow(display_bright)
            ax.set_title(
                f"Original (asset {asset_id})\n"
                f"True: {true_cls_name}\n"
                f"Soft: [{soft_str}]",
                fontsize=8, loc="left"
            )
            ax.axis('off')    
            
            # One column per model
            for col_idx, (run, cam) in enumerate(zip(runs, cams), start=1):
                pred = int(run.preds[idx])
                conf = float(run.probs[idx, pred])
 
                heatmap = cam.compute(input_batch.clone(), pred)[0]
                heat_full = upsample_cam(heatmap, 224)
                overlay = overlay_heatmap(display_bright, heat_full)
 
                status = "correct" if pred == int(hard_i) else "WRONG"
                ax = axes[row_idx, col_idx]
                ax.imshow(np.clip(overlay, 0, 1))
                loss_info = f" [{run.loss_type}]" if run.loss_type else ""
                ax.set_title(
                    f"{run.name}{loss_info}  ({status})\n"
                    f"Predicted: {CLASS_NAMES[pred]}  (conf {conf:.2f})",
                    fontsize=7, loc="left",
                )
                ax.axis("off")
                
    finally:
        for cam in cams:
            cam.close()
 
    fig.suptitle(f"Grad-CAM comparison — {group_name}", fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Wrote {output_path}")

def write_summary(
    groups: dict[str, list[int]], runs: list[RunArtifact], hard: np.ndarray,
    soft: np.ndarray, asset_ids: np.ndarray, output_path: Path
) -> None:
    """Write a plaintext summary of asset_ids in each group"""
    
    lines = ["Grad-CAM selection summary"]
    lines.append(f"Runs ({len(runs)}):")
    
    for r in runs: 
        loss = f"  loss={r.loss_type}" if r.loss_type else ""
        lines.append(f"    {r.name:<40}{loss}")
    lines.append("")
    
    for group_name, indices in groups.items():
        lines.append(f"[{group_name}]  ({len(indices)} examples)")
        if not indices:
            lines.append("  (none)")
            lines.append("")
            continue
        header = f"    {'idx':>6}  {'asset':<10}  {'true':<22}  " + \
            " ".join(f"{r.name[:14]:<14}" for r in runs) + " soft"
            
        lines.append(header)
        
        for idx in indices:
            aid = str(asset_ids[idx])
            true = CLASS_NAMES[int(hard[idx])]
            preds = " ".join(f"{CLASS_NAMES[int(r.preds[idx])]:<14}" for r in runs)
            
            soft_str = " ".join(f"{s:.2f}" for s in soft[idx].tolist())
            lines.append(f"    {idx:>6}  {aid:<10}  {true:<22}  {preds}  [{soft_str}]")
            
        lines.append("")
        
    output_path.write_text("\n".join(lines))
    
    
#################### MAIN ####################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run directories")
    parser.add_argument("--splits-dir", required=True, help="Directory with test.csv and stats.json")
    
    # Image selection - at least one must be provided
    parser.add_argument("--asset-ids", nargs="*", default=None, help="Explicit asset_ids to visualize")
    parser.add_argument("--indices", nargs="*", type=int, default=None,
                        help="Explicit test-set indices to visualize")
    parser.add_argument("--auto-select", action="store_true", help="Auto-pick from categories based on model")
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY, help="Images per auto-category")
    
    # Pathing and options
    parser.add_argument("--image-dir", default="data/gz2/images")
    parser.add_argument("--output-dir", default="output/comparisons/gradcam")
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER, help="Name of model attribute to hook into")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    
    parser.add_argument("--device", default=get_device())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for auto-selection")
    parser.add_argument("--group-name", default="explicit", 
                        help="Name for the output figure when using explicit --asset-ids/--indices")
    
    args = parser.parse_args()
    
    # Validate at least one image specifier is present
    if not args.asset_ids and not args.indices and not args.auto_select:
        parser.error("Provide one or more of: --asset-ids, --indices, --auto-select")
        
    if args.auto_select and len(args.runs) < 2:
        parser.error("--auto-select requires at least 2 --runs for meaningful categories")
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
        
    # Build test loader
    splits_dir = Path(args.splits_dir)
    stats = json.loads((splits_dir / "stats.json").read_text())
    mean = stats["normalization"]["mean"]
    std = stats["normalization"]["std"]
    
    test_ds = GZ2Dataset(
        splits_dir / "test.csv", args.image_dir, transform=build_eval_transform(mean, std)
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
        pin_memory=True
    )
    
    print(f"Test set: {len(test_ds):,} examples")
    
    # Load each run and caych inference
    runs: list[RunArtifact] = []
    ref_hard: np.ndarray | None = None
    ref_soft: np.ndarray | None = None
    
    for run_path_str in args.runs:
        run_dir = Path(run_path_str)
        run_name = run_dir.name
        
        print(f"Loading {run_name}...")
        model, cfg = load_model_from_run(run_dir, checkpoint=args.checkpoint, device=args.device)
        
        print(f"Running inference for {run_name}...")
        probs, hard, soft = run_inference(model, test_loader, args.device)
        
        runs.append(RunArtifact(
            name=run_name,
            model=model,
            probs=probs,
            preds=probs.argmax(axis=1),
            loss_type=cfg.get("loss")
        ))
        
        if ref_hard is None:
            ref_hard, ref_soft = hard, soft
        else:
            assert np.array_equal(hard, ref_hard), "Hard label mismatch across runs"
            assert np.allclose(soft, ref_soft, equal_nan=True), "Soft target mismatch across runs"
            
            
    asset_ids = test_ds.asset_ids.copy()
    
    # Resolve which images to visualize
    groups: dict[str: list[int]] = {}
    
    # Explicit asset_ids and indices are combined
    explicit: list[int] = []
    if args.asset_ids:
        explicit.extend(resolve_asset_ids_to_indices(args.asset_ids, asset_ids))
    if args.indices:
        validate_indices(args.indices, len(test_ds))
        explicit.extend(args.indices)
        
    if explicit:
        seen = set()
        unique = [i for i in explicit if not (i in seen or seen.add(i))]
        groups[args.group_name] = unique
        print(f"\nExplicit group '{args.group_name}' : {len(unique)} indices")
        
    if args.auto_select:
        print("\nAuto-selecting category examples...")
        auto_groups = auto_select_categories(runs, ref_hard, ref_soft, args.per_category, seed= args.seed)
        
        for name, idxs in auto_groups.items():
            if name in groups:
                name = f"auto_{name}"
            groups[name] = idxs
            
    write_summary(groups, runs, ref_hard, ref_soft, asset_ids, output_dir / "summary.txt")
    print(f"\n Wrote {output_dir / 'summary.txt'}")
    
    print(f"\nGenerating Grad-CAM figures...")
    for group_name, indices in groups.items():
        out_png = output_dir / f"{group_name}.png"
        plot_group(
            group_name, indices, runs, ref_hard, ref_soft, asset_ids, test_ds,
            Path(args.image_dir), args.device, args.target_layer, out_png
        )
        
    print(f"\n--- Grad-CAM outputs written to {output_dir} ---")
    
    
if __name__ == "__main__":
    main()