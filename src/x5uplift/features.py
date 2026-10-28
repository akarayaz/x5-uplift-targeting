"""Client level features, built only from information available at the decision moment (CUTOFF).

The result has one row per client (all clients, not only the experiment). Every feature
group is a separate query, so groups can be removed one at a time in the ablation (notebook 07).
"Recent" means the last SHORT_WINDOW_DAYS before the cutoff, "previous" the window before that.
"""

import json

import duckdb
import pandas as pd

from x5uplift.config import (
    AGE_MAX,
    AGE_MIN,
    CATEGORY_MAP_PATH,
    CUTOFF,
    DATA_INTERIM,
    FEATURE_GROUPS_PATH,
    FEATURES_PATH,
    ID_COL,
    N_TOP_CATEGORIES,
    PRODUCTS_CSV,
    SHORT_WINDOW_DAYS,
    X5_FILES,
)

CUT = f"TIMESTAMP '{CUTOFF}'"  # SQL literal for the decision moment
W = SHORT_WINDOW_DAYS


def connect(memory_limit: str = "2GB") -> duckdb.DuckDBPyConnection:
    """Open DuckDB and register clients, train, purchases (parquet) and products (csv) as views.

    The default memory limit suits an 8 GB laptop. When a query needs more, DuckDB spills to a
    git-ignored folder instead of the current working directory.
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    tmp_dir = DATA_INTERIM / ".duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    for name, meta in X5_FILES.items():
        path = (DATA_INTERIM / meta["parquet"]).as_posix()
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    # All ids stay text, numbers are cast explicitly
    con.execute(f"""
        CREATE OR REPLACE VIEW products AS
        SELECT
            product_id, level_1, level_2, level_3, level_4,
            TRY_CAST(segment_id AS DOUBLE)        AS segment_id,
            brand_id, vendor_id,
            TRY_CAST(netto AS DOUBLE)             AS netto,
            TRY_CAST(is_own_trademark AS INTEGER) AS is_own_trademark,
            TRY_CAST(is_alcohol AS INTEGER)       AS is_alcohol
        FROM read_csv('{PRODUCTS_CSV.as_posix()}', all_varchar = true)
    """)
    return con


def create_baskets(con: duckdb.DuckDBPyConnection) -> None:
    """Create a temp table with one row per basket, keyed by (client_id, transaction_id), up to CUTOFF.

    purchase_sum and the point columns repeat on every item row of a basket, so they are
    taken once per basket. Points spent are negative (or zero) in the raw data; ABS makes
    them positive without producing -0.0.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE baskets AS
        SELECT
            client_id,
            transaction_id,
            MIN(transaction_datetime)      AS ts,
            MIN(store_id)                  AS store_id,
            MAX(purchase_sum)              AS basket_sum,
            COUNT(*)                       AS n_items,
            MAX(regular_points_received)   AS reg_points_received,
            MAX(express_points_received)   AS exp_points_received,
            ABS(MIN(regular_points_spent)) AS reg_points_spent,
            ABS(MIN(express_points_spent)) AS exp_points_spent
        FROM purchases
        WHERE transaction_datetime <= {CUT}
        GROUP BY client_id, transaction_id
    """)


def demographics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group 1: age, gender, card tenure and adoption of the points programme."""
    redeemed = f"first_redeem_date <= {CUT}"  # redemptions after the cutoff are unknown at decision time
    return con.execute(f"""
        SELECT
            client_id,
            CASE WHEN age BETWEEN {AGE_MIN} AND {AGE_MAX} THEN age END             AS age,
            CASE WHEN age BETWEEN {AGE_MIN} AND {AGE_MAX} THEN 0 ELSE 1 END        AS age_missing,
            gender,
            DATE_DIFF('day', first_issue_date, {CUT})                              AS days_since_card_issue,
            CASE WHEN {redeemed} THEN 1 ELSE 0 END                                 AS redeemed_before_cutoff,
            CASE WHEN {redeemed} THEN DATE_DIFF('day', first_redeem_date, {CUT}) END AS days_since_first_redeem,
            -- 536 clients redeem seconds before the issue timestamp (same checkout), so clip at 0
            CASE WHEN {redeemed}
                 THEN GREATEST(DATE_DIFF('day', first_issue_date, first_redeem_date), 0) END AS days_issue_to_redeem
        FROM clients
    """).df()


def rfm_trend(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Groups 2 and 3: recency, frequency, spend, and how behaviour changed in the recent window."""
    recent = f"ts > {CUT} - INTERVAL {W} DAY"
    previous = f"ts > {CUT} - INTERVAL {2 * W} DAY AND ts <= {CUT} - INTERVAL {W} DAY"
    return con.execute(f"""
        WITH gaps AS (
            SELECT
                client_id,
                DATE_DIFF('second', LAG(ts) OVER (PARTITION BY client_id ORDER BY ts), ts) / 86400.0 AS gap_days
            FROM baskets
        ),
        gap_stats AS (
            SELECT client_id, AVG(gap_days) AS mean_gap_days, STDDEV_SAMP(gap_days) AS std_gap_days
            FROM gaps
            WHERE gap_days IS NOT NULL
            GROUP BY client_id
        ),
        base AS (
            SELECT
                client_id,
                COUNT(*)                                           AS n_baskets,
                SUM(basket_sum)                                    AS total_spend,
                AVG(basket_sum)                                    AS avg_basket_spend,
                MAX(basket_sum)                                    AS max_basket_spend,
                STDDEV_SAMP(basket_sum)                            AS std_basket_spend,
                DATE_DIFF('second', MAX(ts), {CUT}) / 86400.0       AS days_since_last_purchase,
                DATE_DIFF('second', MIN(ts), {CUT}) / 86400.0       AS days_since_first_purchase,
                COUNT(*) FILTER (WHERE {recent})                   AS n_baskets_recent,
                COUNT(*) FILTER (WHERE {previous})                 AS n_baskets_previous,
                COALESCE(SUM(basket_sum) FILTER (WHERE {recent}), 0)   AS spend_recent,
                COALESCE(SUM(basket_sum) FILTER (WHERE {previous}), 0) AS spend_previous
            FROM baskets
            GROUP BY client_id
        )
        SELECT
            b.*,
            g.mean_gap_days,
            g.std_gap_days,
            -- above 1: the client is later than usual for the next visit
            b.days_since_last_purchase / NULLIF(g.mean_gap_days, 0)          AS overdue_ratio,
            b.spend_recent / NULLIF(b.total_spend, 0)                        AS share_spend_recent,
            -- -1 = only shopped in the previous window, +1 = only in the recent one
            CAST(b.n_baskets_recent - b.n_baskets_previous AS DOUBLE)
                / NULLIF(b.n_baskets_recent + b.n_baskets_previous, 0)       AS basket_trend
        FROM base AS b
        LEFT JOIN gap_stats AS g USING (client_id)
    """).df()


def points(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group 4: regular and express loyalty points received and spent."""
    return con.execute("""
        SELECT
            client_id,
            SUM(reg_points_received) AS reg_points_received,
            SUM(reg_points_spent)    AS reg_points_spent,
            SUM(exp_points_received) AS exp_points_received,
            SUM(exp_points_spent)    AS exp_points_spent,
            AVG(CASE WHEN reg_points_spent > 0 OR exp_points_spent > 0 THEN 1.0 ELSE 0.0 END) AS share_baskets_points_spent,
            AVG(CASE WHEN exp_points_received > 0 THEN 1.0 ELSE 0.0 END)                    AS share_baskets_express_received
        FROM baskets
        GROUP BY client_id
    """).df()


def basket_content(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group 5: basket size, product variety and how often the same products are bought again."""
    return con.execute(f"""
        WITH product_in_basket AS (
            -- DISTINCT as its own step: DuckDB can spill it to disk, unlike COUNT(DISTINCT) inside GROUP BY
            SELECT DISTINCT client_id, product_id, transaction_id
            FROM purchases
            WHERE transaction_datetime <= {CUT}
        ),
        client_product AS (
            SELECT client_id, product_id, COUNT(*) AS n_baskets_with_product
            FROM product_in_basket
            GROUP BY client_id, product_id
        ),
        variety AS (
            SELECT
                client_id,
                COUNT(*) AS n_distinct_products,
                AVG(CASE WHEN n_baskets_with_product >= 2 THEN 1.0 ELSE 0.0 END) AS share_products_repeated
            FROM client_product
            GROUP BY client_id
        ),
        size AS (
            SELECT client_id, AVG(n_items) AS avg_items_per_basket
            FROM baskets
            GROUP BY client_id
        )
        SELECT s.client_id, s.avg_items_per_basket, v.n_distinct_products, v.share_products_repeated
        FROM size AS s
        JOIN variety AS v USING (client_id)
    """).df()


def stores(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group 6: number of stores used and loyalty to the most used store."""
    return con.execute("""
        WITH client_store AS (
            SELECT client_id, store_id, COUNT(*) AS n
            FROM baskets
            GROUP BY client_id, store_id
        )
        SELECT
            client_id,
            COUNT(*)                   AS n_distinct_stores,
            MAX(n) * 1.0 / SUM(n)      AS top_store_share
        FROM client_store
        GROUP BY client_id
    """).df()


def time_habits(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group 7: weekend and evening shopping. DAYOFWEEK: Sunday = 0, Saturday = 6."""
    return con.execute("""
        SELECT
            client_id,
            AVG(CASE WHEN DAYOFWEEK(ts) IN (0, 6) THEN 1.0 ELSE 0.0 END) AS share_weekend_baskets,
            AVG(CASE WHEN HOUR(ts) >= 18 THEN 1.0 ELSE 0.0 END)          AS share_evening_baskets
        FROM baskets
        GROUP BY client_id
    """).df()

def category_ranking(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """level_2 categories ranked by total spend over all clients (the target is not used).

    cat_rank 1 is the largest category. Category ids are hashed in the data, so features are
    named by rank and this table maps each rank back to its id.
    """
    ranking = con.execute(f"""
        SELECT p.level_2, SUM(u.trn_sum_from_iss) AS spend
        FROM purchases AS u
        JOIN products AS p USING (product_id)
        WHERE u.transaction_datetime <= {CUT} AND p.level_2 IS NOT NULL
        GROUP BY p.level_2
        ORDER BY spend DESC, p.level_2
    """).df()
    ranking["cat_rank"] = range(1, len(ranking) + 1)
    ranking["spend_share"] = ranking["spend"] / ranking["spend"].sum()
    return ranking


def categories(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Group 8: spend shares of the top level_2 categories (the rest as 'other'), own brand and
    alcohol shares, and how many level_3 categories the client buys from.

    Item amounts (trn_sum_from_iss) are used because the basket total is not split by item.
    """
    top = category_ranking(con).head(N_TOP_CATEGORIES)[["level_2", "cat_rank"]]
    con.register("top_categories", top)
    share = "/ NULLIF(SUM(i.amount), 0)"
    top_cols = ",\n            ".join(
        f"SUM(CASE WHEN t.cat_rank = {r} THEN i.amount ELSE 0 END) {share} AS share_cat_{r:02d}"
        for r in top["cat_rank"]
    )
    # Step 1 shrinks 45.8M item rows to one row per client and level_2 category.
    # Step 2 counts level_3 categories with a separate DISTINCT, which DuckDB can spill to disk.
    return con.execute(f"""
        WITH client_cat AS (
            SELECT
                u.client_id,
                p.level_2,
                SUM(u.trn_sum_from_iss)                                          AS amount,
                SUM(CASE WHEN p.is_own_trademark = 1 THEN u.trn_sum_from_iss ELSE 0 END) AS own_amount,
                SUM(CASE WHEN p.is_alcohol = 1 THEN u.trn_sum_from_iss ELSE 0 END)       AS alcohol_amount
            FROM purchases AS u
            JOIN products AS p USING (product_id)
            WHERE u.transaction_datetime <= {CUT}
            GROUP BY u.client_id, p.level_2
        ),
        shares AS (
            SELECT
                i.client_id,
                {top_cols},
                SUM(CASE WHEN t.cat_rank IS NULL THEN i.amount ELSE 0 END) {share} AS share_cat_other,
                SUM(i.own_amount) {share}                                          AS share_own_brand,
                SUM(i.alcohol_amount) {share}                                      AS share_alcohol
            FROM client_cat AS i
            LEFT JOIN top_categories AS t USING (level_2)
            GROUP BY i.client_id
        ),
        client_level3 AS (
            SELECT DISTINCT u.client_id, p.level_3
            FROM purchases AS u
            JOIN products AS p USING (product_id)
            WHERE u.transaction_datetime <= {CUT} AND p.level_3 IS NOT NULL
        ),
        variety AS (
            SELECT client_id, COUNT(*) AS n_distinct_categories
            FROM client_level3
            GROUP BY client_id
        )
        SELECT s.*, v.n_distinct_categories
        FROM shares AS s
        LEFT JOIN variety AS v USING (client_id)
    """).df()

FEATURE_GROUPS = {
    "demographics": demographics,
    "rfm_trend": rfm_trend,
    "points": points,
    "basket_content": basket_content,
    "stores": stores,
    "time_habits": time_habits,
    "categories": categories,
}


def build_features(con: duckdb.DuckDBPyConnection, groups: dict = FEATURE_GROUPS) -> tuple[pd.DataFrame, dict]:
    """Build every group and join them on the client id.

    Returns the feature table and a {group name: [column names]} map used by the ablation.
    """
    create_baskets(con)
    table = None
    columns = {}
    for name, build in groups.items():
        part = build(con)
        columns[name] = [c for c in part.columns if c != ID_COL]
        if table is None:
            table = part
        else:
            table = table.merge(part, on=ID_COL, how="left", validate="one_to_one")
    return table, columns


def save_features(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    """Build all features and save them, together with the category map and the group map."""
    table, columns = build_features(con)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURES_PATH, index=False)
    category_ranking(con).to_csv(CATEGORY_MAP_PATH, index=False)
    FEATURE_GROUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_GROUPS_PATH.write_text(json.dumps(columns, indent=2))
    return table, columns