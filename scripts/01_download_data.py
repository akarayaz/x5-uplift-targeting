"""Download the three X5 RetailHero files without loading them into memory.

scikit-uplift's fetch_x5 reads purchases.csv.gz fully into pandas (about 45.8 million
rows), so the files are downloaded directly here and parsed later by DuckDB.
Source URLs and MD5 hashes come from the scikit-uplift 0.5.1 source (sklift.datasets).
The raw files stay in data/raw and are never committed.
"""

import argparse
import hashlib
import sys
import time
import urllib.request

from x5uplift.config import DATA_RAW, X5_BASE_URL, X5_FILES

CHUNK = 4 * 1024 * 1024


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    tmp = dest.parent / (dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} of {total / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:8.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def check_hash(dest, expected):
    actual = md5_of(dest)
    if actual == expected:
        print(f"  MD5 ok ({actual})")
        return True
    print(
        f"  WARNING: MD5 {actual} differs from the recorded {expected}.\n"
        "  Either the download is incomplete or the recorded hash is wrong. The file is kept."
    )
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name, meta in X5_FILES.items():
        dest = DATA_RAW / meta["gz"]
        url = X5_BASE_URL + meta["gz"]
        print(f"{name}: {dest.name}")
        if dest.exists() and not args.force:
            print("  already present, verifying")
        else:
            start = time.time()
            try:
                download(url, dest)
            except Exception as exc:  # network, certificate or disk problems
                print(f"\n  Download failed: {exc!r}")
                print(f"  Fallback: curl -L -o {dest.as_posix()} {url}")
                all_ok = False
                continue
            print(f"  done in {time.time() - start:.0f} s")
        all_ok &= check_hash(dest, meta["md5"])

    print("\nFiles in data/raw:")
    for p in sorted(DATA_RAW.glob("*.gz")):
        print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")
    print("The documentation lists about 647 MB compressed in total.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
