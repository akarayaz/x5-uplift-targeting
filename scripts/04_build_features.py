"""Build the client feature table from the parquet files.

Writes data/processed/client_features.parquet (one row per client, information known at the
decision moment only), data/processed/category_map.csv and reports/feature_groups.json.
The feature definitions live in src/x5uplift/features.py.
"""

import sys
import time

from x5uplift.features import connect, save_features


def main():
    start = time.time()
    con = connect()
    table, groups = save_features(con)
    print(f"{table.shape[0]:,} clients x {table.shape[1] - 1} features in {time.time() - start:.0f} s")
    for group, cols in groups.items():
        print(f"  {group}: {len(cols)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
