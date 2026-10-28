"""Convert the raw gzipped CSV files to Parquet with DuckDB.

DuckDB streams the CSV, so the 45.8 million row purchases file is never held in memory
as a whole. Column types are auto detected. If a value does not fit the detected type,
DuckDB raises an error instead of silently changing it. The id column is forced to
VARCHAR so that joins across files cannot fail on type differences.
"""

import argparse
import sys
import time

import duckdb

from x5uplift.audit_utils import lit
from x5uplift.config import DATA_INTERIM, DATA_RAW, ID_COL, X5_FILES


def convert(con, src, dst, id_col):
    src_sql = lit(src.as_posix())
    described = con.execute(f"DESCRIBE SELECT * FROM read_csv({src_sql})").fetchall()
    names = [r[0] for r in described]
    types = f", types={{{lit(id_col)}: 'VARCHAR'}}" if id_col in names else ""
    if not types:
        print(f"  note: column {id_col!r} not found, columns are {names}")

    tmp = dst.with_name(dst.name + ".tmp")
    sql = (
        f"COPY (SELECT * FROM read_csv({src_sql}{types})) TO {lit(tmp.as_posix())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
    )
    start = time.time()
    con.execute(sql)
    tmp.replace(dst)

    dst_sql = lit(dst.as_posix())
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet({dst_sql})").fetchone()[0]
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({dst_sql})").fetchall()
    print(f"  rows: {n:,}   parquet size: {dst.stat().st_size / 1e6:.1f} MB   time: {time.time() - start:.0f} s")
    for col, typ, *_ in schema:
        print(f"    {col}: {typ}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memory-limit", default="4GB", help="DuckDB memory limit, for example 4GB")
    ap.add_argument("--force", action="store_true", help="convert again if the parquet file exists")
    ap.add_argument("--only", nargs="*", choices=list(X5_FILES), help="convert only these tables")
    ap.add_argument("--id-col", default=ID_COL)
    args = ap.parse_args()

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_INTERIM / ".duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit={lit(args.memory_limit)}")
    con.execute(f"SET temp_directory={lit(tmp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order=false")

    for name in args.only or list(X5_FILES):
        meta = X5_FILES[name]
        src = DATA_RAW / meta["gz"]
        dst = DATA_INTERIM / meta["parquet"]
        print(f"{name}: {src.name} -> {dst.name}")
        if not src.exists():
            print(f"  missing {src}, run scripts/01_download_data.py first")
            return 1
        if dst.exists() and not args.force:
            print("  parquet exists, skipping (use --force to redo)")
            continue
        convert(con, src, dst, args.id_col)
    return 0


if __name__ == "__main__":
    sys.exit(main())
