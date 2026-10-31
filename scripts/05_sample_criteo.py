"""Draw a reproducible random sample of the Criteo uplift dataset (v2.1) for notebook 07.

Download the raw file first (about 311 MB):
    curl -L -o data/raw/criteo-research-uplift-v2.1.csv.gz <CRITEO_URL in config.py>
The full file has about 14 million rows; it is read in chunks so memory stays low, and each row is
kept with probability CRITEO_SAMPLE_SHARE using a fixed seed. Raw data is never committed.
"""

import sys
import time

import numpy as np
import pandas as pd

from x5uplift.config import CRITEO_RAW, CRITEO_SAMPLE_PATH, CRITEO_SAMPLE_SHARE, SEED


def main():
    if not CRITEO_RAW.exists():
        print(f"Missing {CRITEO_RAW}; download it first (see the docstring).")
        return 1
    start = time.time()
    rng = np.random.default_rng(SEED)
    parts, total = [], 0
    for chunk in pd.read_csv(CRITEO_RAW, chunksize=1_000_000, dtype="float32"):
        total += len(chunk)
        parts.append(chunk[rng.random(len(chunk)) < CRITEO_SAMPLE_SHARE])
    sample = pd.concat(parts, ignore_index=True)
    labels = ["treatment", "exposure", "visit", "conversion"]
    sample[labels] = sample[labels].astype("int8")
    CRITEO_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(CRITEO_SAMPLE_PATH, index=False)
    print(f"{total:,} rows read, {len(sample):,} kept ({len(sample) / total:.2%}) in {time.time() - start:.0f} s")
    print(sample[labels].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
