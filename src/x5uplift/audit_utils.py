"""Small helpers shared by the audit script and later notebooks."""

import math
from datetime import date, datetime
from decimal import Decimal

from scipy import stats

_NUMERIC = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE", "DECIMAL",
}
_TEMPORAL = {
    "DATE", "TIME", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS",
}


def qi(name: str) -> str:
    """Quote an identifier for DuckDB SQL."""
    return '"' + name.replace('"', '""') + '"'


def lit(value) -> str:
    """Quote a string literal for DuckDB SQL, for example a file path."""
    return "'" + str(value).replace("'", "''") + "'"


def base_type(duck_type: str) -> str:
    """DECIMAL(18,3) becomes DECIMAL."""
    return duck_type.split("(")[0].strip().upper()


def is_numeric_type(duck_type: str) -> bool:
    return base_type(duck_type) in _NUMERIC


def is_temporal_type(duck_type: str) -> bool:
    return base_type(duck_type) in _TEMPORAL


def smd(mean_t: float, var_t: float, mean_c: float, var_c: float) -> float:
    """Standardized mean difference: (mean_t - mean_c) / sqrt((var_t + var_c) / 2).

    Returns NaN when the pooled variance is missing or zero.
    """
    if var_t is None or var_c is None or mean_t is None or mean_c is None:
        return float("nan")
    pooled = math.sqrt((var_t + var_c) / 2)
    if pooled == 0:
        return float("nan")
    return (mean_t - mean_c) / pooled


def srm_chisquare(n_treatment: int, n_control: int, expected_treatment_share: float = 0.5):
    """Sample ratio mismatch test: chi square goodness of fit against the design split.

    Returns (chi2, p_value). A very small p value means the observed group sizes are
    unlikely under the intended assignment ratio.
    """
    n = n_treatment + n_control
    exp_t = n * expected_treatment_share
    exp_c = n * (1 - expected_treatment_share)
    chi2 = (n_treatment - exp_t) ** 2 / exp_t + (n_control - exp_c) ** 2 / exp_c
    return chi2, float(stats.chi2.sf(chi2, df=1))


def clean(obj):
    """Make an object JSON friendly: NaN and inf to None, Decimal to float, dates to text."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, Decimal):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        return clean(obj.item())
    return obj
