"""Data audit for the X5 RetailHero parquet files.

Research question: what do the files actually contain, are customer ids consistent
across files, and does the treatment assignment look randomized?

Only three column names come from the documentation (id, treatment, target, see
x5uplift.config). Every other column is discovered from the files. The console report is
for reading. reports/data_audit.json keeps aggregated results only, never customer rows.
"""

import argparse
import json
import sys

import duckdb
import pandas as pd
from scipy import stats

from x5uplift.audit_utils import (
    base_type,
    clean,
    is_numeric_type,
    is_temporal_type,
    lit,
    qi,
    smd,
    srm_chisquare,
)
from x5uplift.config import (
    DATA_INTERIM,
    EXPECTED_TREATMENT_SHARE,
    ID_COL,
    REPORTS,
    TARGET_COL,
    TREATMENT_COL,
    X5_FILES,
)

SMD_FLAG = 0.1
MAX_LEVELS = 50


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def rel(name):
    path = DATA_INTERIM / X5_FILES[name]["parquet"]
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/02_convert_to_parquet.py first.")
    return f"read_parquet({lit(path.as_posix())})"


def columns_of(con, relation):
    rows = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    return [(r[0], r[1]) for r in rows]


def timestamp_share(con, relation, col):
    """Share of non null values in a 10,000 row sample that parse as a timestamp."""
    c = qi(col)
    sql = (
        f"SELECT COUNT(TRY_CAST({c} AS TIMESTAMP)) * 1.0 / NULLIF(COUNT({c}), 0) "
        f"FROM (SELECT {c} FROM {relation} USING SAMPLE 10000 ROWS)"
    )
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Schema and column profiles
# ---------------------------------------------------------------------------
def profile_table(con, name, id_col, approx_distinct):
    relation = rel(name)
    n = con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    out = {"rows": n, "columns": []}
    for cname, ctype in columns_of(con, relation):
        c = qi(cname)
        distinct = f"approx_count_distinct({c})" if approx_distinct else f"COUNT(DISTINCT {c})"
        select = f"COUNT(*) - COUNT({c}), {distinct}"
        kind = "other"
        if is_numeric_type(ctype):
            kind = "numeric"
            select += f", MIN({c}), MAX({c}), AVG({c}), STDDEV_SAMP({c})"
        elif is_temporal_type(ctype):
            kind = "temporal"
            select += f", MIN({c}), MAX({c})"
        row = con.execute(f"SELECT {select} FROM {relation}").fetchone()
        info = {
            "name": cname,
            "type": ctype,
            "kind": kind,
            "nulls": row[0],
            "null_share": row[0] / n if n else 0.0,
            "distinct": row[1],
            "distinct_is_approximate": approx_distinct,
        }
        if kind == "numeric":
            info.update(min=row[2], max=row[3], mean=row[4], std=row[5])
        elif kind == "temporal":
            info.update(min=row[2], max=row[3])
        elif base_type(ctype) == "VARCHAR" and cname != id_col:
            share = timestamp_share(con, relation, cname)
            info["timestamp_parse_share_in_sample"] = share
            if share is not None and share >= 0.99:
                mn, mx = con.execute(
                    f"SELECT MIN(TRY_CAST({c} AS TIMESTAMP)), MAX(TRY_CAST({c} AS TIMESTAMP)) FROM {relation}"
                ).fetchone()
                info["parsed_min"], info["parsed_max"] = mn, mx
        out["columns"].append(info)

    print(f"\n[{name}] rows: {n:,}" + ("   (distinct counts are approximate)" if approx_distinct else ""))
    table = pd.DataFrame(
        [
            {
                "column": c["name"],
                "type": c["type"],
                "null_share": round(c["null_share"], 4),
                "distinct": c["distinct"],
                "min": c.get("min", c.get("parsed_min")),
                "max": c.get("max", c.get("parsed_max")),
                "ts_parse": c.get("timestamp_parse_share_in_sample"),
            }
            for c in out["columns"]
        ]
    )
    print(table.to_string(index=False))
    head_cols = [c for c, _ in columns_of(con, relation)]
    exclude = f" EXCLUDE ({qi(id_col)})" if id_col in head_cols else ""
    print("first 5 rows (id column hidden):")
    print(con.execute(f"SELECT *{exclude} FROM {relation} LIMIT 5").df().to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# 2. Id consistency across files
# ---------------------------------------------------------------------------
def id_consistency(con, id_col):
    names = {n: [c for c, _ in columns_of(con, rel(n))] for n in X5_FILES}
    has = {n: id_col in cols for n, cols in names.items()}
    out = {"id_col": id_col, "id_present": has}
    q = qi(id_col)
    one = lambda sql: con.execute(sql).fetchone()[0]

    for n in ("clients", "train"):
        if has[n]:
            rows, distinct, nulls = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT {q}), COUNT(*) - COUNT({q}) FROM {rel(n)}"
            ).fetchone()
            out[f"{n}_rows"], out[f"{n}_distinct_ids"], out[f"{n}_null_ids"] = rows, distinct, nulls
            out[f"{n}_duplicate_rows"] = rows - distinct

    if has["train"] and has["clients"]:
        out["train_ids_missing_in_clients"] = one(
            f"SELECT COUNT(*) FROM {rel('train')} t WHERE NOT EXISTS "
            f"(SELECT 1 FROM {rel('clients')} c WHERE c.{q} = t.{q})"
        )
    if has["purchases"]:
        con.execute(f"CREATE OR REPLACE TEMP TABLE pid AS SELECT DISTINCT {q} AS _id FROM {rel('purchases')}")
        out["purchases_distinct_ids"] = one("SELECT COUNT(*) FROM pid")
        if has["clients"]:
            out["purchase_ids_missing_in_clients"] = one(
                f"SELECT COUNT(*) FROM pid WHERE NOT EXISTS (SELECT 1 FROM {rel('clients')} c WHERE c.{q} = pid._id)"
            )
            out["clients_without_purchases"] = one(
                f"SELECT COUNT(*) FROM {rel('clients')} c WHERE NOT EXISTS (SELECT 1 FROM pid WHERE pid._id = c.{q})"
            )
        if has["train"]:
            out["train_clients_without_purchases"] = one(
                f"SELECT COUNT(*) FROM {rel('train')} t WHERE NOT EXISTS (SELECT 1 FROM pid WHERE pid._id = t.{q})"
            )
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


# ---------------------------------------------------------------------------
# 3. Treatment assignment: group sizes, target rates, sample ratio mismatch
# ---------------------------------------------------------------------------
def experiment_check(con, treat_col, target_col, expected_share):
    relation = rel("train")
    cols = dict(columns_of(con, relation))
    missing = [c for c in (treat_col, target_col) if c not in cols]
    if missing:
        print(f"  Columns not found in train: {missing}. Columns present: {list(cols)}")
        return {"error": f"missing columns {missing}", "columns": list(cols)}
    tq, yq = qi(treat_col), qi(target_col)
    out = {"treatment_type": cols[treat_col], "target_type": cols[target_col]}

    tv = con.execute(f"SELECT {tq}, COUNT(*) FROM {relation} GROUP BY 1 ORDER BY 1").fetchall()
    yv = con.execute(f"SELECT {yq}, COUNT(*) FROM {relation} GROUP BY 1 ORDER BY 1 LIMIT 20").fetchall()
    out["treatment_value_counts"] = {str(k): n for k, n in tv}
    out["target_value_counts_first_20"] = {str(k): n for k, n in yv}
    print(f"  treatment values ({cols[treat_col]}): {out['treatment_value_counts']}")
    print(f"  target values ({cols[target_col]}), up to 20: {out['target_value_counts_first_20']}")

    if not {k for k, _ in tv} <= {0, 1}:
        print("  Treatment is not a clean 0/1 flag (null or other values). Group analysis skipped.")
        return out

    groups = con.execute(
        f"SELECT {tq}, COUNT(*), COUNT({yq}), AVG(CAST({yq} AS DOUBLE)) FROM {relation} GROUP BY 1 ORDER BY 1"
    ).fetchall()
    by = {int(k): (n, nn, mean) for k, n, nn, mean in groups}
    if 0 not in by or 1 not in by:
        print("  Only one treatment group present. Group analysis skipped.")
        return out
    (n_c, nn_c, y_c), (n_t, nn_t, y_t) = by[0], by[1]
    overall = con.execute(f"SELECT AVG(CAST({yq} AS DOUBLE)) FROM {relation}").fetchone()[0]
    chi2, p = srm_chisquare(n_t, n_c, expected_share)
    out.update(
        control_n=n_c,
        treatment_n=n_t,
        treatment_share=n_t / (n_t + n_c),
        expected_treatment_share=expected_share,
        srm_chi2=chi2,
        srm_p_value=p,
        target_nulls_control=n_c - nn_c,
        target_nulls_treatment=n_t - nn_t,
        target_mean_control=y_c,
        target_mean_treatment=y_t,
        target_mean_overall=overall,
    )
    print(f"  control:   n = {n_c:,}   target mean = {y_c:.4f}")
    print(f"  treatment: n = {n_t:,}   target mean = {y_t:.4f}")
    print(f"  overall target mean = {overall:.4f}  (documented response ratio: 0.62, definition not given)")
    print(f"  treatment share = {n_t / (n_t + n_c):.4f}  (design assumption {expected_share})")
    print(f"  sample ratio mismatch test: chi2 = {chi2:.3f}, p = {p:.4f}")
    return out


# ---------------------------------------------------------------------------
# 4. Balance of pre contact covariates between treatment and control
# ---------------------------------------------------------------------------
def balance_check(con, id_col, treat_col):
    cc = dict(columns_of(con, rel("clients")))
    tc = dict(columns_of(con, rel("train")))
    pc = dict(columns_of(con, rel("purchases")))
    if not (id_col in cc and id_col in tc and id_col in pc and treat_col in tc):
        print("  Skipped: the id or treatment column is missing in one of the files.")
        return {"skipped": True}
    q, tq = qi(id_col), qi(treat_col)
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE pr AS SELECT {q} AS _id, COUNT(*) AS purchase_rows "
        f"FROM {rel('purchases')} GROUP BY 1"
    )
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE bal AS "
        f"SELECT t.{tq} AS _treat, c.* EXCLUDE ({q}), COALESCE(pr.purchase_rows, 0) AS purchase_rows "
        f"FROM {rel('train')} t "
        f"LEFT JOIN {rel('clients')} c ON t.{q} = c.{q} "
        f"LEFT JOIN pr ON t.{q} = pr._id"
    )

    numeric_rows, categorical_rows, skipped = [], [], []
    for name, typ in columns_of(con, "bal"):
        if name == "_treat":
            continue
        c = qi(name)
        expr = None
        if is_numeric_type(typ):
            expr = f"CAST({c} AS DOUBLE)"
        elif is_temporal_type(typ):
            expr = f"epoch(CAST({c} AS TIMESTAMP))"
        elif base_type(typ) == "VARCHAR":
            share = timestamp_share(con, "bal", name)
            if share is not None and share >= 0.99:
                expr = f"epoch(TRY_CAST({c} AS TIMESTAMP))"

        if expr is not None:
            rows = con.execute(
                f"SELECT _treat, COUNT(*), COUNT({expr}), AVG({expr}), VAR_SAMP({expr}) "
                f"FROM bal GROUP BY 1 ORDER BY 1"
            ).fetchall()
            by = {int(k): v for k, *v in rows if k is not None}
            if 0 in by and 1 in by:
                (n0, nn0, m0, v0), (n1, nn1, m1, v1) = by[0], by[1]
                numeric_rows.append(
                    {
                        "column": name,
                        "mean_control": m0,
                        "mean_treatment": m1,
                        "smd": smd(m1, v1, m0, v0),
                        "nonnull_share_control": nn0 / n0,
                        "nonnull_share_treatment": nn1 / n1,
                    }
                )
        elif base_type(typ) in ("VARCHAR", "BOOLEAN"):
            levels = con.execute(f"SELECT COUNT(DISTINCT {c}) FROM bal").fetchone()[0]
            if 2 <= levels <= MAX_LEVELS:
                ct = con.execute(
                    f"SELECT _treat, CAST({c} AS VARCHAR) AS v, COUNT(*) AS n FROM bal "
                    f"WHERE {c} IS NOT NULL GROUP BY 1, 2"
                ).df()
                table = ct.pivot(index="v", columns="_treat", values="n").fillna(0)
                chi2, p, dof, _ = stats.chi2_contingency(table.values)
                categorical_rows.append({"column": name, "levels": levels, "chi2": chi2, "dof": dof, "p_value": p})
            else:
                skipped.append({"column": name, "levels": levels})

    out = {"numeric": numeric_rows, "categorical": categorical_rows, "skipped_columns": skipped}
    if numeric_rows:
        df = pd.DataFrame(numeric_rows)
        df = df.reindex(df["smd"].abs().sort_values(ascending=False, na_position="last").index)
        print("  Numeric and date covariates, sorted by absolute standardized mean difference:")
        print(df.round(4).to_string(index=False))
        out["max_abs_smd"] = float(df["smd"].abs().max())
        out["n_above_flag"] = int((df["smd"].abs() > SMD_FLAG).sum())
        print(f"  largest |SMD| = {out['max_abs_smd']:.4f}; columns above {SMD_FLAG}: {out['n_above_flag']}")
    if categorical_rows:
        df = pd.DataFrame(categorical_rows).sort_values("p_value")
        print(f"\n  Categorical covariates (chi square test of independence with treatment, {len(df)} tests):")
        print(df.round(4).to_string(index=False))
        print(f"  Bonferroni threshold for these tests at 0.05: {0.05 / len(df):.4f}")
    if skipped:
        print(f"\n  Not tested (more than {MAX_LEVELS} levels or not parseable): {skipped}")
    print("  purchase_rows is the number of rows per customer in purchases (a row is not assumed to be a basket).")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id-col", default=ID_COL)
    ap.add_argument("--treatment-col", default=TREATMENT_COL)
    ap.add_argument("--target-col", default=TARGET_COL)
    ap.add_argument("--expected-treatment-share", type=float, default=EXPECTED_TREATMENT_SHARE)
    ap.add_argument("--memory-limit", default="4GB")
    args = ap.parse_args()

    tmp_dir = DATA_INTERIM / ".duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit={lit(args.memory_limit)}")
    con.execute(f"SET temp_directory={lit(tmp_dir.as_posix())}")

    report = {"duckdb_version": duckdb.__version__, "profiles": {}}

    section("1. Schema and column profiles")
    for name in ("clients", "train", "purchases"):
        report["profiles"][name] = profile_table(con, name, args.id_col, approx_distinct=(name == "purchases"))

    section("2. Customer id consistency across files")
    report["id_consistency"] = id_consistency(con, args.id_col)

    section("3. Treatment assignment in the labeled subset (train)")
    report["experiment"] = experiment_check(
        con, args.treatment_col, args.target_col, args.expected_treatment_share
    )

    section("4. Balance of pre contact covariates")
    report["balance"] = balance_check(con, args.id_col, args.treatment_col)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "data_audit.json"
    with open(out_path, "w") as f:
        json.dump(clean(report), f, indent=2)
    print(f"\nAggregated audit written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
