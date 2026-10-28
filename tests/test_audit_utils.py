import math
from decimal import Decimal

from scipy import stats

from x5uplift.audit_utils import clean, is_numeric_type, lit, qi, smd, srm_chisquare


def test_smd_with_unit_variances():
    assert math.isclose(smd(1.0, 1.0, 0.0, 1.0), 1.0)


def test_smd_zero_variance_is_nan():
    assert math.isnan(smd(1.0, 0.0, 0.0, 0.0))


def test_srm_matches_scipy():
    chi2, p = srm_chisquare(10200, 10000, 0.5)
    ref = stats.chisquare([10200, 10000], [10100, 10100])
    assert math.isclose(chi2, ref.statistic)
    assert math.isclose(p, ref.pvalue)


def test_srm_perfect_split():
    chi2, p = srm_chisquare(500, 500, 0.5)
    assert chi2 == 0
    assert math.isclose(p, 1.0)


def test_quoting():
    assert qi('a"b') == '"a""b"'
    assert lit("it's") == "'it''s'"


def test_type_helpers():
    assert is_numeric_type("DECIMAL(18,3)")
    assert not is_numeric_type("VARCHAR")


def test_clean():
    assert clean({"a": float("nan"), "b": Decimal("1.5"), "c": [float("inf")]}) == {
        "a": None,
        "b": 1.5,
        "c": [None],
    }
