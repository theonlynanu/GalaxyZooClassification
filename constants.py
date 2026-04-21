"""
Danyal Ahmed - April 2026

constants.py
Defines constants to be used throughout the project for analysis and config
"""


###################### CONSTANTS ######################

####        Hyperparameters and Tunable Constants       ####

# Number of classes (not really tunable)
N_CLASSES = 4

# Default Thresholds
THRESHOLDS = {
    "smooth": 0.7,
    "edgeon": 0.7,
    "spiral": 0.7
}

OOD_SPLIT_COL = "REDSHIFT"
OOD_SPLIT_DEFAULT = 70      # percentile

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

####        Class Definitions       ####
CLASS_NAMES = {
    0: "Elliptical",
    1: "Edge-on disk",
    2: "Face-on spiral",
    3: "Face-on non-spiral"
}

CLASS_SOFT_COL = {
    0: COL_SMOOTH,
    1: COL_EDGEON,
    2: COL_SPIRAL,
    3: COL_NOSPIRAL
}

CLASS_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


####        Settings/Configuration       ####
DEFAULT_SEED = 112568       # 112568 doesn't come from anywhere in particular, I just hit keys on my keyboard