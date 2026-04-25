# GalaxyZooClassification

A modular pipeline for galaxy morphology classification on the GalaxyZoo2 dataset, supporting analysis of model behavior under domain shift, the effects of covariates on model behavior, comparison of CNNs with linear SVM and/or Gradient-Boosted Decision Trees, and the effects of hard and soft label training regimes for CNNs.

This repository provides scripts for the full pipeline: data joining and labeling, train/val/test splitting, classical and CNN feature extraction, model training, evaluation, and Grad-CAM visualization.

## Repository layout
Note that not all of these directories will be immediately available upon cloning this repository. The starred (*) directories are those created as default names for the various programs, which you may name differently with the appropriate flags, but take care to pass the non-default directory names to any subsequently-run programs.

```
GalaxyZooClassification/
├── README.md
├── requirements.txt
├── constants.py               # Column names, thresholds, class definitions, seed
├── dataset.py                 # PyTorch Dataset + transform builders
├── analysis.py                # Dataset diagnostics + label construction
├── prepare_splits.py          # Stratified train/val/test splits + normalization stats
├── feature_extraction.py      # 207-dim engineered features for classical models
├── train.py                   # CNN training loop (CE or KL loss)
├── train_classical.py         # SVM and GBDT training + evaluation
├── evaluate.py                # CNN evaluation (calibration, soft-target metrics)
├── evaluate_domain.py         # Cross-model domain-shift evaluation
├── extra_domain_eval.py       # Supplementary domain-shift analyses
├── gradcam.py                 # Grad-CAM visualizations
├── models/
│   ├── __init__.py
│   └── basic_cnn.py
├── data/
│   └── gz2/
│       ├── sample_images/     # One hundred sample .jpg cutouts
│       ├── images/            # ~240k .jpg cutouts
│       ├── raw/               # Hart16 + GZ2 sample + filename mapping CSVs
│       └── processed/
│           └── splits/
│               ├── easyhard/  # *easy.csv, hard.csv, split_meta.txt
│               ├── domain/    # *train/val/test for domain-shift project
│               └── soft/      # *train/val/test for label-regime project
└── output/           
    ├── runs/                  # *per-model training runs (checkpoints, logs, plots)
    ├── comparisons/           # *multi-run comparisons (evaluate.py, gradcam.py)
    └── domain_eval/           # *cross-model domain-shift outputs
```

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

The PyTorch install in `requirements.txt` does not pin a CUDA build. If GPU acceleration is needed, install the appropriate PyTorch build for your hardware first (see [PyTorch install selector](https://pytorch.org/get-started/locally/)).

### Data

The pipeline expects three GalaxyZoo2 CSVs and the full image archive. None are committed to the repository.

| File                       | Source                                     | Link                                                                                                                                                                                    |
| -------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gz2_hart16.csv`           | Hart et al. (2016) debiased vote fractions | [GZ2 (Table 1)](https://data.galaxyzoo.org/#section-7) ([Direct Download](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz))                                                          |
| `gz2sample.csv`            | GZ2 sample metadata (redshift, photometry) | [GZ2 (Bottom table labeled 'SDSS metadata for GZ2')](https://data.galaxyzoo.org/#section-7) ([Direct Download](https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/gz2sample.csv.gz)) |
| `gz2_filename_mapping.csv` | object ID → image asset mapping            | [Zenodo (official source)](https://zenodo.org/records/3565489) ([Direct Download](https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1))[]                      |
| Image cutouts (~5 GB)      | GZ2 Zenodo image release                   | [Zenodo (official source)](https://zenodo.org/records/3565489) ([Direct Download](https://zenodo.org/records/3565489/files/images_gz2.zip?download=1))                                  |

Place the 3 CSVs under `data/gz2/raw/` and the `.jpg` images under `data/gz2/images/`.

If you do not wish to download the full image dataset, you may move the 100 sample images from `data/gz2/sample_images/` into `data/gz2/images/`. The collection of the CSVs is necessary, however.

## Pipeline

The full pipeline runs in five stages: data inventory and labeling, splitting, feature extraction (classical only), training, and evaluation. Each script is independent and re-runnable. Defaults assume the directory layout above; everything important is overridable via flags.

### 1. Data inventory and labeling — `analysis.py`

Joins the three raw CSVs, audits image availability, applies vote-fraction thresholds to assign hard labels, and writes:

- `data/gz2/processed/gz2_labeled.csv` — full labeled dataset
- `data/gz2/processed/splits/easyhard/easy.csv` and `hard.csv` — domain split at the 70th percentile of redshift
- `data/gz2/processed/splits/easyhard/split_meta.txt` — split metadata

Also produces dataset-level diagnostic plots (vote distributions, redshift histograms, class balance, etc.) when invoked with `--save` (which I recommend), allowing you to determine hyperparameter selection yourself if you so choose.


```bash
python analysis.py \
    data/gz2/raw/gz2_hart16.csv \
    data/gz2/raw/gz2sample.csv \
    data/gz2/raw/gz2_filename_mapping.csv \
    data/gz2/processed/gz2_labeled.csv \
    --save
```
<img width="1200" alt="sample_image_grid" src="https://github.com/user-attachments/assets/896b5bcb-dcba-4e6e-be37-01db7def4b2d" />

*One of the sample image grids created during analysis, depicting sample images from each class*

<br>

Full syntax:

```bash
python analysis.py <gz2_hart16.csv> <gz2samples.csv> <mapping.csv> [out.csv] [--save]
```

### 2. Train/val/test splits — `prepare_splits.py`

````

Creates a stratified-sampled 70/15/15 split, with per-channel image normalization statistics computed from the training fold.

The primary behavioral modes are:

- **Standard:** input is `gz2_labeled.csv` (full labeled set), saves thresholded labels to `data/gz2/processed/full` using `data/gz2/processed/gz2_labeled.csv` and `data/gz2/images/`.
- **Domain-shift analysis (`--domain-split`):** input should be `easy.csv` (low-redshift only), so models are trained on the easy domain and then evaluated on `hard.csv` as out-of-domain.
- **Label/loss analysis (`--soft-targets`):** compute and store branch-product soft targets alongside the hard labels, for KL-loss training.

```bash
# Standard
python prepare_splits.py

# Domain-shift analysis: low-z subset only
python prepare_splits.py \
    data/gz2/processed/splits/easyhard/easy.csv \
    data/gz2/images \
    data/gz2/processed/splits/domain \
    --domain-split

# Label/loss analysis: request soft target data
python prepare_splits.py \
    data/gz2/processed/gz2_labeled.csv \
    data/gz2/images \
    data/gz2/processed/splits/soft \
    --soft-targets
````

Outputs `train.csv`, `val.csv`, `test.csv`, and `stats.json` (normalization mean/std) in the chosen output directory. The following syntax block of this section also demonstrates the ability to vary the program behavior, such as the total sample size, or the number of training images used to compute normalization stats; run `python prepare_splits.py -h` to see these options more clearly.

Full syntax:
```bash
python prepare_splits.py [labeled_csv] [image_dir] [out_dir] [-h] [--size SIZE] [--seed SEED] [--stats-sample STATS_SAMPLE] [--soft-targets] [--domain-split]
```

### 3. Feature extraction (classical models only) — `feature_extraction.py`

Computes a 207-dimensional feature vector per image: scalar shape statistics (concentration, asymmetry, smoothness, Gini, M20, ellipticity, position angle, normalized Petrosian radius), a 10-bin radial flux profile, and a three-level spatial pyramid gradient histogram. Multi-process by default. Writes one `.npz` per split.

```bash
python feature_extraction.py \
    --splits-dir data/gz2/processed/splits/domain \
    --ood-csv data/gz2/processed/splits/easyhard/hard.csv \
    --workers 8
```

Outputs `features_train.npz`, `features_val.npz`, `features_test.npz`, and `features_ood.npz` in the splits directory inside of the provided output directory.

Run `python feature_extraction.py` for a full breakdown of extra options available for further tuning.

Full syntax:

```bash
usage: feature_extraction.py --splits-dir SPLITS_DIR [--ood-csv OOD_CSV] [--image-dir IMAGE_DIR] [--out-dir OUT_DIR] [--splits {train,val,test,ood} [{train,val,test,ood} ...]] [--crop-size CROP_SIZE] [--profile-bins PROFILE_BINS] [--gradient-bins GRADIENT_BINS] [--workers WORKERS] [--no-verify] [-h]
```

### 4. Training

#### CNN — `train.py`

Trains the BasicCNN architecture (defined in `models/basic_cnn.py`) end-to-end. Supports both cross-entropy (standard hard labels) and Kullback-Leibler divergence (soft targets) losses. Saves checkpoint at best validation loss plus the final epoch.

```bash
# eg: Hard-label CE on domain-split data
python train.py --loss ce \
    --splits-dir data/gz2/processed/splits/domain \
    --epochs 30 --batch-size 128 --lr 3e-4

# eg: Soft-label KL on the full set
python train.py --loss kl \
    --splits-dir data/gz2/processed/splits/soft \
    --epochs 30
```

Note that KL divergence loss cannot be run on a data split created by `prepare_splits.py` without the `--soft-targets` flag. 

Run output goes to `output/runs/<auto-generated-name>/`, including:

- `config.json` — training configuration
- `checkpoint_best.pt`, `checkpoint_last.pt`
- `training_log.csv` — per-epoch metrics
- `plots/` — loss, accuracy, and per-class curves

<img width="1000" alt="accuracy_curves" src="https://github.com/user-attachments/assets/c9a4bc5f-b804-449e-b0db-406c9c258e2b" />

*An example training performance graph*

<br>

Full syntax (run `python train.py -h` for a more clear breakdown):
```bash
train.py [-h] --loss {ce,kl} --splits-dir SPLITS_DIR [--image-dir IMAGE_DIR] [--output-root OUTPUT_ROOT] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--weight-decay WEIGHT_DECAY] [--dropout DROPOUT]
                [--eta-min ETA_MIN] [--seed SEED] [--num-workers NUM_WORKERS] [--device DEVICE] [--force]
```

#### Classical models — `train_classical.py`

Linear SVM (`LinearSVC`) or histogram-based gradient-boosted trees (`HistGradientBoostingClassifier`), trained on the engineered feature vectors. Performs grid search on validation, refits on train+val, evaluates on test and OOD.

```bash
python train_classical.py --model svm \
    --features-dir data/gz2/processed/splits/domain \
    --output-root output/runs

python train_classical.py --model gbdt \
    --features-dir data/gz2/processed/splits/domain
```

Run output includes `model.joblib`, `metrics.json` (IID + OOD performance, per-class accuracy, confusion matrices), `predictions_test.npz`, `predictions_ood.npz`, and grid search log.

The classical models do not currently support soft-label targets, and must be trained on a data split created without the `--soft-targets` flag.

### 5. Evaluation

#### CNN evaluation — `evaluate.py`

Evaluates one or more CNN runs on a shared test set, producing calibration curves, ECE, Brier score, KL-to-soft-target divergence, and ambiguity-stratified breakdowns.

If provided multiple models after `--runs`, it will provide comparative data, including grouped bar charts, summary data in each .csv file, and confusion matrices for each model.

<img width="500" alt="accuracy_by_group" src="https://github.com/user-attachments/assets/03df7456-4b79-4f09-997f-8855b2e8ba4e" /> <img width="500" alt="accuracy_by_group" src="https://github.com/user-attachments/assets/148aaf77-7c81-484e-87b8-91cf57f6641f" />

*Two example charts showing ECE by stratum, one from a single-model evaluation and the other a comparative evaluation*

<br>

```bash
# Basic one-model run. Default output to output/comparisons
python evaluate.py \
    --runs output/runs/full_ce_lr3e-4 output/runs/soft_ce_lr3e-4 \
    --splits-dir data/gz2/processed/splits/domain
        # Note the split for testing does not need to be the same used to train the model!

python evaluate.py \
    --runs output/runs/soft_kl_lr3e-4 output/runs/soft_ce_lr3e-4 \
    --splits-dir data/gz2/processed/splits/soft \
    --output-dir output/comparisons/soft_vs_hard
```

#### Cross-model domain-shift evaluation — `evaluate_domain.py`

Evaluates any subset of {SVM, GBDT, CNN} on both the IID test set and the OOD hard set. Produces a metrics CSV, summary table, confusion matrices, and redshift-binned performance plots with bootstrap CIs.

```bash
# All three model types
python evaluate_domain.py --models svm gbdt cnn \
    --svm-run output/runs/domain_svm \
    --gbdt-run output/runs/domain_gbdt \
    --cnn-run output/runs/domain_ce_lr3e-4_bs128_e30 \
    --output-dir output/domain_eval

# Only evaluating svm
python evaluate_domain.py --models svm \
    --svm-run output/runs/domain_svm \
    --output-dir output/svm
```

Per-model flags are independent — pass any subset of `--models` and only those backends will be loaded.

#### Supplementary domain-shift analyses — `extra_domain_eval.py`

Per-class precision/recall/F1 by redshift bin, linear-fit degradation slopes, and a controlled IID-vs-OOD comparison at matched redshift. Same CLI as `evaluate_domain.py`:

```bash
python extra_domain_eval.py \
    --svm-run output/runs/domain_svm \
    --gbdt-run output/runs/domain_gbdt \
    --cnn-run output/runs/domain_ce_lr3e-4_bs128_e30 \
    --output-dir output/domain_eval
```

### Grad-CAM visualizations — `gradcam.py`

Generates per-image saliency overlays for one or more trained CNN runs. Can target specific images by `--asset-ids` or `--indices`, or auto-select a class-balanced grid via `--auto-select`. Auto-selection automatically selects a set of 4 test-set images meeting the following criteria: 
- Every model agrees *and* predicts the correct class
- Every model predicts incorrectly (they may disagree on their predicted incorrect classes)
- Model disagreement is very high
- The image has an ambiguous original vote distribution

<img height="750" alt="report_figure" src="https://github.com/user-attachments/assets/dc2c6121-0680-4e7a-b175-6d74fa802a87" />

*An auto-generated Grad-CAM visualization between a model trained with cross-entropy loss and another with Kullback-Leibler*

<br>

```bash
# Auto-select 4 images per class for two runs
python gradcam.py \
    --runs output/runs/soft_kl_lr3e-4 output/runs/soft_ce_lr3e-4 \
    --splits-dir data/gz2/processed/splits/soft \
    --auto-select --per-category 4 \
    --output-dir output/comparisons/gradcam
```

## Module reference

| Module                  | Purpose                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| `constants.py`          | Column names, vote-fraction thresholds, class definitions, default random seed |
| `dataset.py`            | `GZ2Dataset`, transform builders, weighted sampler, OOD loader factory         |
| `models/basic_cnn.py`   | `BasicCNN` architecture (~1.87M params)                                        |
| `analysis.py`           | Three-CSV join, image audit, threshold sweeps, label assignment, dataset plots |
| `prepare_splits.py`     | Stratified split, normalization stats, optional soft-target computation        |
| `feature_extraction.py` | Engineered feature pipeline for classical models                               |
| `train.py`              | CNN training (CE/KL loss, cosine annealing, weighted sampling)                 |
| `train_classical.py`    | SVM/GBDT training with grid search, IID + OOD evaluation                       |
| `evaluate.py`           | CNN-only evaluation, calibration analysis, soft-target metrics                 |
| `evaluate_domain.py`    | Cross-model IID vs OOD evaluation, redshift binning                            |
| `extra_domain_eval.py`  | Supplementary precision/recall, slopes, controlled comparison                  |
| `gradcam.py`            | Grad-CAM overlays for trained CNN runs                                         |

## Notes

- By default, all scripts honor a fixed random seed (`DEFAULT_SEED = 112568` in `constants.py`) for reproducibility. This can be changed or set to random by changing the field in constants.py
- `dataset.py` falls back to one-hot soft labels if the CSV has no `soft_k` columns, so the same `GZ2Dataset` works for both label-regime and domain-shift data.
- Model checkpoints, training logs, evaluation outputs, and the `data/` directory are gitignored by default. See `.gitignore` for the precise rules.
- Most scripts have a `--help` flag that documents all options; the commands above show common defaults but not exhaustive flag lists.

## References

- Willett et al., "Galaxy Zoo 2: detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey," MNRAS, 2013.
- Hart et al., "Galaxy Zoo: comparing the demographics of spiral arm number and a new method for correcting redshift bias," MNRAS, 2016.
- Dieleman, Willett, Dambre, "Rotation-invariant convolutional neural networks for galaxy morphology prediction," MNRAS, 2015.

## Author

Danyal Ahmed — Northeastern University, Khoury College of Computer Sciences
