"""Central configuration: seed, paths, column names and source file metadata."""

import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

# Column names as given in the scikit-uplift documentation. They are re-checked against
# the real files by scripts/03_inspect_data.py. Nothing else about the schema is assumed.
ID_COL = "client_id"
TREATMENT_COL = "treatment_flg"
TARGET_COL = "target"

# Decision moment: the last timestamp in the purchase history (checked in notebook 01).
# Any model input must be known at or before this moment.
CUTOFF = "2019-03-18 23:40:03"

# Valid age range (decided in notebook 01). Values outside it are treated as missing.
AGE_MIN = 14
AGE_MAX = 100

# Labeled data split (decided in notebook 01): untouched test share and cross-fitting folds.
TEST_SHARE = 0.30
N_FOLDS = 5
SPLITS_PATH = DATA_PROCESSED / "splits.parquet"

# Client features (decided in notebook 01): short window for recent behaviour, output file.
SHORT_WINDOW_DAYS = 30
FEATURES_PATH = DATA_PROCESSED / "client_features.parquet"

# Category features (decided in notebook 02): top level_2 categories by spend get their own share.
N_TOP_CATEGORIES = 12
CATEGORY_MAP_PATH = DATA_PROCESSED / "category_map.csv"
FEATURE_GROUPS_PATH = REPORTS / "feature_groups.json"

# Client segments (decided in notebook 02): k-means with k = 7, saved per client.
N_SEGMENTS = 7
SEGMENTS_PATH = DATA_PROCESSED / "segments.parquet"

# Purchase probability without an SMS (notebook 03), used by the policies in notebook 06.
PROPENSITY_PATH = DATA_PROCESSED / "propensity.parquet"

# Documented treatment share (scikit-uplift page: "Treatment ratio 0.5"). This is a design
# assumption used for the sample ratio mismatch test, not a measured fact.
EXPECTED_TREATMENT_SHARE = 0.5

# products.csv is not in the scikit-uplift files. It comes from the official competition
# archive (ods.ai, retailhero-uplift.zip, data/products.csv), extracted by hand into data/raw.
PRODUCTS_CSV = DATA_RAW / "products.csv"

# Source: scikit-uplift 0.5.1, sklift.datasets.datasets (x5_metadata), read via its
# documentation page. The MD5 values were transcribed from that page. If all three show a
# mismatch after a complete download, suspect the recorded hashes before suspecting the files.
X5_BASE_URL = "https://sklift.s3.eu-west-2.amazonaws.com/"
X5_FILES = {
    "clients": {
        "gz": "clients.csv.gz",
        "md5": "b9cdeb2806b732771de03e819b3354c5",
        "parquet": "clients.parquet",
    },
    "train": {
        "gz": "uplift_train.csv.gz",
        "md5": "2720bbb659daa9e0989b2777b6a42d19",
        "parquet": "uplift_train.parquet",
    },
    "purchases": {
        "gz": "purchases.csv.gz",
        "md5": "48d2de13428e24e8b61d66fef02957a8",
        "parquet": "purchases.parquet",
    },
}


def set_seed(seed: int = SEED) -> None:
    """Seed python and numpy. Models and splits should also receive the seed explicitly."""
    random.seed(seed)
    np.random.seed(seed)
# Out-of-fold uplift scores of the training folds (notebook 05), saved so slow models run only once.
UPLIFT_SCORES_PATH = DATA_PROCESSED / "uplift_scores.parquet"

# Predicted uplift of the chosen model (notebook 05) for all experiment clients: out of fold for the
# training folds, from one model fitted on all training folds for the test set. Used in notebook 06.
UPLIFT_PATH = DATA_PROCESSED / "uplift.parquet"

# Aggregated outputs of notebook 06 read by the Streamlit demo (no client level data; safe to publish).
DEMO_DATA_DIR = ROOT / "app" / "data"

# Second randomized dataset (notebook 07): Criteo uplift v2.1, Criteo AI Lab, mirrored on Hugging Face
# (datasets/criteo/criteo-uplift), license CC BY-NC-SA 4.0. Raw file stays local; a 10% sample is used.
CRITEO_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
CRITEO_RAW = DATA_RAW / "criteo-research-uplift-v2.1.csv.gz"
CRITEO_SAMPLE_PATH = DATA_PROCESSED / "criteo_sample.parquet"
CRITEO_SAMPLE_SHARE = 0.10
