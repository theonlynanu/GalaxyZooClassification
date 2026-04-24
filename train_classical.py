"""
Danyal Ahmed - April 2026

train_classical.py
Train one of the two classical ML models (Linear SVM or Gradient Boosted Trees)
on the crafted feature vectors from feature_extraction.py

Pipeline:
    1) Load features_train.npz, features_val.npz, features_test.npz, features_ood.npz
    2) Fit StandardScaler on training features
    3) Grid search over hyperparameters, selecting by validation macro-F1
    4) Refit best model on combined train/val
    5) Evaluate on IID test and OOD test
    6) Save model, scaler, config, predictions, metrics, and training log

Output directory structure:
    output/runs/<run_name>/
        model.joblib            fitted classifier
        scaler.joblib           fitted StandardScaler   
        config.json             hyperparameters and metadata
        predictions_test.npz    IID test predictions
        predictions_ood.npz     OOD test predictions
        metrics.json            macro-F1 and per-class accuracy on both test sets
        training_log.txt        human (me) - readable log
        
Usage:
    python train_classical.py --model svm
    python train_classical.py --model gbdt
    python train_classical.py --model svm --features-dir data/dz2/processed/splits/domain    
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix

from constants import CLASS_NAMES, N_CLASSES, DEFAULT_SEED


#################### CONSTANTS ####################

DEFAULT_FEATURES_DIR = Path("data/gz2/processed/splits/domain")
DEFAULT_OUTPUT_ROOT = Path("output/runs")


####                Grid search spaces              ####

SVM_GRID = [
    {"C": 0.01},
    {"C": 0.1},
    {"C": 1.0},
    {"C": 10.0},
]

GBDT_GRID = [
    {"learning_rate": 0.05, "max_iter": 200},
    {"learning_rate": 0.05, "max_iter": 400},
    {"learning_rate": 0.10, "max_iter": 200},
    {"learning_rate": 0.10, "max_iter": 400},
]


#################### DATA HANDLING ####################

@dataclass
class RunConfig:
    """All hyperparameters and metadata for one classical model run"""
    model: str
    features_dir: str
    output_dir: str
    best_hyperparams: dict
    grid_results: list
    seed: int
    n_train: int
    n_val: int
    n_test: int
    n_ood: int
    n_features: int
    n_dropped_nan: dict     # per-split count of dropped NaN rows
    
    
def load_features(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load a feature.npz and drop NaN rows

    Args:
        npz_path (Path): path to features_{split}.npz

    Returns:
        (
            X: (n, n_features) feature matrix with NaN rows removed
            y: (n,) integer labels
            objids: (n,) dr7objid values for alignment checks
            n_dropped (int): number of rows dropped due to NaN error
        )
    """
    data = np.load(npz_path)
    X = data["X"]
    y = data["labels"]
    objids = data["objids"]
    
    nan_mask = np.isnan(X).any(axis=1)
    n_dropped = int(nan_mask.sum())
    
    if n_dropped > 0:
        print(f"    Dropped {n_dropped} rows with NaN features from {npz_path.name}")
        X = X[~nan_mask]
        y = y[~nan_mask]
        objids = objids[~nan_mask]
        
    return X.astype(np.float32), y.astype(np.int64), objids.astype(np.int64), n_dropped


def make_run_name(args) -> str:
    """Build a run directory name from odel and features directory stem"""
    
    features_stem = Path(args.features_dir).stem
    return f"{features_stem}_{args.model}"


#################### MODEL CONSTRUCTION ####################

def build_model(model_name: str, hyperparams: dict, seed: int):
    """Build a scikit-learn model with the given hyperparameters

    Args:
        model_name (str): "svm" | "gbdt"
        hyperparams (dict): model-specific hyperparameters
        seed (int): random seed
    """
    if model_name == "svm":
        return LinearSVC(
            C = hyperparams["C"],
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            dual="auto"     # picks primal vs dual based on n vs p
        )
    
    if model_name == "gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=hyperparams["learning_rate"],
            max_iter=hyperparams["max_iter"],
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            verbose=1
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_grid(model_name: str) -> list[dict]:
    """Return the hyperparameter grid for the given model"""
    
    if model_name == "svm":
        return SVM_GRID
    if model_name == "gbdt":
        return GBDT_GRID
    raise ValueError(f"Unknown model: {model_name}")


def grid_search(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, seed: int
) -> tuple[dict, list[dict]]:
    """Sweep the hyperparameter grid and select the best by validation macro-F1

    Args:
        model_name (str): "svm" or "gbdt"
        X_train (np.ndarray): scaled training features
        y_train (np.ndarray): training labels
        X_val (np.ndarray): scaled validation features
        y_val (np.ndarray): validation labels
        seed (int): random seed

    Returns:
        tuple[dict, list[dict]]: 
            (
                best_hyperparams(dict): hyperparameters with highest validation macro-F1
                results (list[dict]): all grid combinations with their validation scores
            )
    """
    grid = get_grid(model_name)
    results = []
    
    print(f"\nGrid search over {len(grid)} combinations:")
    print(f"    {'hyperparams':<35}  {'val macro-F1':>12}  {'val accuracy':>12}  {'fit time':>10}")
    print(f"   " + "-" * 74)
    
    for hp in grid:
        t0 = time.time()
        model = build_model(model_name, hp, seed)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average="macro")
        accuracy = (preds == y_val).mean()
        fit_time = time.time() - t0
        
        results.append({
            "hyperparams": hp,
            "val_macro_f1": float(macro_f1),
            "val_accuracy": float(accuracy),
            "fit_seconds": float(fit_time)
        })
        
        print(f"    {str(hp):<35}  {macro_f1:>12.4f}  {accuracy:>12.4f}  {fit_time:>9.1f}s")
        
    # Select best by validation macro-F1
    best_idx = int(np.argmax([r["val_macro_f1"] for r in results]))
    best = results[best_idx]
    
    print(f"\n  Best: {best['hyperparams']}  val_macro_f1={best['val_macro_f1']:.4f}")
    
    return best["hyperparams"], results


#################### EVALUATION ####################

def evaluate_split(
    model, X: np.ndarray, y: np.ndarray, split_name: str
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Run the model on a test split and compute metrics

    Args:
        model: fitted classifier
        X (np.ndarray): scaled features
        y (np.ndarray): true labels
        split_name (str): "iit" or "ood", for log output

    Returns:
        tuple[np.ndarray, np.ndarray | None, dict]:
            (
                predicted labels, predicted probabilities if available, 
                {"macro_f1", "accuracy", "per-class accuracy", "confusion matrix"}
            )
    """
    
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    
    macro_f1 = f1_score(y, preds, average="macro")
    accuracy = (preds == y).mean()
    cm = confusion_matrix(y, preds, labels=list(range(N_CLASSES)))
    
    per_class_acc = {}
    for k in range(N_CLASSES):
        mask = y == k
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[k]] = float((preds[mask] == k).mean())
        else:
            per_class_acc[CLASS_NAMES[k]] = 0.0
            
    metrics = {
        "macro_f1":     float(macro_f1),
        "accuracy":     float(accuracy),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y))
    }
    
    print(f"\n{split_name.upper()} evaluation:")
    print(f"    n = {len(y):,}")
    print(f"    macro F1  = {macro_f1:.4f}")
    print(f"    accuracy  = {accuracy:.4f}")
    print(f"    per-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f"        {name:<22}  {acc:.4f}")

    return preds, probs, metrics


#################### OUTPUT WRITING ####################

def save_predictions(out_path: Path, objids: np.ndarray, y_true: np.ndarray,
                     y_pred: np.ndarray, probs: np.ndarray | None) -> None:
    """Save predictions to .npz for later comparison"""
    
    payload = {
        "objids": objids,
        "true": y_true,
        "pred": y_pred
    }
    
    if probs is not None:
        payload["probs"] = probs
        
    np.savez_compressed(out_path, **payload)    # Pointers in python always feels so weird
    

def write_training_log(
    log_path: Path, cfg: RunConfig, iid_metrics: dict, ood_metrics: dict
) -> None:
    """Human readable training summary for the run"""
    lines = []
    lines = []
    lines.append("=" * 72)
    lines.append(f"Classical model training: {cfg.model.upper()}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Features dir:  {cfg.features_dir}")
    lines.append(f"Output dir:    {cfg.output_dir}")
    lines.append(f"Seed:          {cfg.seed}")
    lines.append(f"n_train:       {cfg.n_train:,}")
    lines.append(f"n_val:         {cfg.n_val:,}")
    lines.append(f"n_test (IID):  {cfg.n_test:,}")
    lines.append(f"n_test (OOD):  {cfg.n_ood:,}")
    lines.append(f"n_features:    {cfg.n_features}")    
    
    if any(v > 0 for v in cfg.n_dropped_nan.values()):
        lines.append("")
        lines.append("NaN rows dropped per split:")
        for split, n in cfg.n_dropped_nan.items():
            lines.append(f"    {split:<8}: {n}")
            
    lines.append("")
    lines.append("-" * 72)
    lines.append("Grid search results")
    lines.append("-" * 72)
    lines.append(f"{'hyperparams':<35}  {'val macro-F1':>12}  {'val accuracy':>12}")
    for r in cfg.grid_results:
        marker = " *" if r["hyperparams"] == cfg.best_hyperparams else "  "
        lines.append(f"{str(r['hyperparams']):<35}  "
                     f"{r['val_macro_f1']:>12.4f}  "
                     f"{r['val_accuracy']:>12.4f}{marker}")
    lines.append("")
    lines.append(f"Best hyperparameters: {cfg.best_hyperparams}")

    for split_name, metrics in [("IID test", iid_metrics), ("OOD test", ood_metrics)]:
        lines.append("")
        lines.append("-" * 72)
        lines.append(f"{split_name} evaluation (best model, refit on train+val)")
        lines.append("-" * 72)
        lines.append(f"n         = {metrics['n']:,}")
        lines.append(f"macro F1  = {metrics['macro_f1']:.4f}")
        lines.append(f"accuracy  = {metrics['accuracy']:.4f}")
        lines.append("Per-class accuracy:")
        for name, acc in metrics["per_class_accuracy"].items():
            lines.append(f"    {name:<22}  {acc:.4f}")

        lines.append("")
        lines.append("Confusion matrix (rows = true class, columns = predicted):")
        cm     = np.array(metrics["confusion_matrix"])
        header = "                         " + "  ".join(f"{CLASS_NAMES[k][:8]:>8}" for k in range(N_CLASSES))
        lines.append(header)
        for i in range(N_CLASSES):
            row = f"    {CLASS_NAMES[i]:<22} " + "  ".join(f"{cm[i, j]:>8}" for j in range(N_CLASSES))
            lines.append(row)

    lines.append("")
    lines.append("=" * 72)
    log_path.write_text("\n".join(lines))
    
    
#################### MAIN ####################

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--model", required=True, choices=["svm", "gbdt"],
                        help="Which classical model to train")
    
    # Paths
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR,
                        help=f"Directory containing features_{{split}}.npz (default: {DEFAULT_FEATURES_DIR})")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Parent directory for output folders (default: {DEFAULT_OUTPUT_ROOT})")
    
    # Options
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite existing run directory without prompt")
    
    args = parser.parse_args()
    run_name = make_run_name(args)
    run_dir = args.output_root / run_name
    
    if run_dir.exists() and not args.force:
        print(f"WARNING: {run_dir} already exists. Overwrite?")
        while True:
            answer = input("[y/n] ")
            if answer.lower() in ["y", "yes"]:
                break
            elif answer.lower() in ["n", "no"]:
                print("Exiting.")
                sys.exit(0)
            else:
                print("Unknown input.")
                
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Model:         {args.model.upper()}")
    print(f"Features dir:  {args.features_dir}")
    print(f"Seed:          {args.seed}")

    # Load features
    print(f"\nLoading features from {args.features_dir}...")

    X_train, y_train, _, n_drop_train = load_features(args.features_dir / "features_train.npz")
    X_val, y_val, _, n_drop_val = load_features(args.features_dir / "features_val.npz")
    X_test, y_test,  obj_test, n_drop_test = load_features(args.features_dir / "features_test.npz")
    X_ood, y_ood, obj_ood, n_drop_ood = load_features(args.features_dir / "features_ood.npz")
    
    n_features = X_train.shape[1]
    print(f"\n    n_train = {len(X_train):,}  "
          f"n_val = {len(X_val):,}  "
          f"n_test = {len(X_test):,}  "
          f"n_ood = {len(X_ood):,}")
    print(f"    n_features = {n_features}")

    # Fit scaler on training only
    print("\nFitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    X_ood_s   = scaler.transform(X_ood)

    # Grid search 
    best_hp, grid_results = grid_search(
        args.model, X_train_s, y_train, X_val_s, y_val, args.seed
    )

    # Refit best model on train + val
    print(f"\nRefitting {args.model.upper()} with {best_hp} on train + val...")

    X_refit = np.concatenate([X_train_s, X_val_s], axis=0)
    y_refit = np.concatenate([y_train,   y_val  ], axis=0)

    t0          = time.time()
    final_model = build_model(args.model, best_hp, args.seed)
    final_model.fit(X_refit, y_refit)
    print(f"    refit time: {time.time() - t0:.1f}s")

    # Evaluate on IID and OOD 
    iid_pred, iid_probs, iid_metrics = evaluate_split(final_model, X_test_s, y_test, "iid")
    ood_pred, ood_probs, ood_metrics = evaluate_split(final_model, X_ood_s,  y_ood,  "ood")

    # IID vs OOD degradation
    degradation = iid_metrics["macro_f1"] - ood_metrics["macro_f1"]
    print(f"\nDegradation (IID → OOD): macro-F1 drops by {degradation:.4f}")

    # Save artifacts - my first time using joblib so I hope this works
    print(f"\nSaving artifacts to {run_dir}...")

    joblib.dump(final_model, run_dir / "model.joblib")
    joblib.dump(scaler,      run_dir / "scaler.joblib")

    save_predictions(run_dir / "predictions_test.npz",
                     obj_test, y_test, iid_pred, iid_probs)
    save_predictions(run_dir / "predictions_ood.npz",
                     obj_ood,  y_ood,  ood_pred, ood_probs)

    cfg = RunConfig(
        model            = args.model,
        features_dir     = str(args.features_dir),
        output_dir       = str(run_dir),
        best_hyperparams = best_hp,
        grid_results     = grid_results,
        seed             = args.seed,
        n_train          = len(X_train),
        n_val            = len(X_val),
        n_test           = len(X_test),
        n_ood            = len(X_ood),
        n_features       = n_features,
        n_dropped_nan    = {
            "train": n_drop_train, "val": n_drop_val,
            "test":  n_drop_test,  "ood": n_drop_ood,
        },
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    metrics_payload = {
        "iid": iid_metrics,
        "ood": ood_metrics,
        "degradation_macro_f1": float(degradation),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    # Final summary
    write_training_log(run_dir / "training_log.txt", cfg, iid_metrics, ood_metrics)

    print(f"\n--- Summary ({args.model.upper()}) ---")
    print(f"Run directory:  {run_dir}")
    print(f"Best params:    {best_hp}")
    print(f"IID macro-F1:   {iid_metrics['macro_f1']:.4f}")
    print(f"OOD macro-F1:   {ood_metrics['macro_f1']:.4f}")
    print(f"Degradation:    {degradation:.4f}")


if __name__ == "__main__":
    main()
    
