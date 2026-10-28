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

# Documented treatment share (scikit-uplift page: "Treatment ratio 0.5"). This is a design
# assumption used for the sample ratio mismatch test, not a measured fact.
EXPECTED_TREATMENT_SHARE = 0.5

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
