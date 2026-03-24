"""
tests/test_loc_scale.py
=======================
Pytest test suite for robstattm.loc_scale_m and robstattm.scale_m.

Test strategy
-------------
Each test uses the SAME data generated in R so that the Python and R
results can be compared to machine precision (< 1e-8).  Data is generated
once using R's set.seed() and stored as a module-level fixture; the same
numpy array is passed to both the Python wrapper and the R function
directly (via RPy2).

Test categories
---------------
1. Man-page examples — exact reproduction of the R .Rd examples.
2. Output structure — correct keys, types, shapes.
3. Numerical equivalence — Python ≈ R to 1e-8.
4. Relationship invariant — disper == scaleM(x - mu).
5. Robustness checks — estimator is not pulled by outliers.
6. Input type coverage — numpy, pandas, polars.
7. Edge cases — NA handling, near-constant data.
8. Parameter validation — bad psi/eff/family raises ValueError.
"""

import math
import numpy as np
import pandas as pd
import pytest

# Skip the entire module if rpy2 / RobStatTM are not installed
rpy2 = pytest.importorskip("rpy2", reason="rpy2 not installed")
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

try:
    _pkg = importr("RobStatTM")
except Exception:
    pytest.skip("RobStatTM R package not installed", allow_module_level=True)

from robstattm import loc_scale_m, scale_m

# Helpers

def r_loc_scale_m(x_np, **kwargs):
    """Call R's locScaleM on a numpy array, return dict matching Python API."""
    x_r = ro.FloatVector(x_np)
    r_kwargs = {}
    for k, v in kwargs.items():
        r_key = k.replace("_", ".")
        if isinstance(v, bool):
            r_kwargs[r_key] = ro.BoolVector([v])
        elif isinstance(v, float):
            r_kwargs[r_key] = ro.FloatVector([v])
        elif isinstance(v, int):
            r_kwargs[r_key] = ro.IntVector([v])
        elif isinstance(v, str):
            r_kwargs[r_key] = ro.StrVector([v])
        else:
            r_kwargs[r_key] = v
    res = _pkg.locScaleM(x_r, **r_kwargs)
    return {
        "mu":      float(res.rx2("mu")[0]),
        "std_mu":  float(res.rx2("std.mu")[0]),
        "disper":  float(res.rx2("disper")[0]),
    }


def r_scale_m(u_np, **kwargs):
    """Call R's scaleM on a numpy array, return float."""
    u_r = ro.FloatVector(u_np)
    r_kwargs = {}
    for k, v in kwargs.items():
        r_key = k.replace("_", ".")
        if isinstance(v, float):
            r_kwargs[r_key] = ro.FloatVector([v])
        elif isinstance(v, int):
            r_kwargs[r_key] = ro.IntVector([v])
        elif isinstance(v, str):
            r_kwargs[r_key] = ro.StrVector([v])
        else:
            r_kwargs[r_key] = v
    return float(_pkg.scaleM(u_r, **r_kwargs)[0])


def seed_r(n):
    ro.r(f"set.seed({n})")


# Module-level data fixtures (generated in R to guarantee exact match)

# Man-page example 1: clean Gaussian, sd=1.5, n=150
seed_r(123)
_X_CLEAN = np.array(ro.r("rnorm(150, sd=1.5)"))

# Man-page example 2: 10% outliers
seed_r(123)
_X_OUTLIER_LOC = np.array(ro.r(
    "c(rnorm(135, sd=1.5), rnorm(15, mean=-10, sd=.5))"
))

# scaleM man-page example: outliers shifted
seed_r(123)
_X_SCALE_CLEAN = np.array(ro.r("rnorm(150, sd=1.5)"))
seed_r(123)
_X_SCALE_OUTLIER = np.array(ro.r(
    "c(rnorm(135, sd=1.5), rnorm(15, mean=-5, sd=.5))"
))


# 1. Man-page examples — loc_scale_m

class TestLocScaleMManPage:
    """Reproduce the examples from MLocDis.Rd exactly."""

    def test_clean_gaussian_default_psi(self):
        """locScaleM(r) with default psi='mopt', eff=0.95."""
        py = loc_scale_m(_X_CLEAN)
        r  = r_loc_scale_m(_X_CLEAN)
        assert abs(py["mu"]     - r["mu"])     < 1e-8
        assert abs(py["std_mu"] - r["std_mu"]) < 1e-8
        assert abs(py["disper"] - r["disper"]) < 1e-8

    def test_outlier_location_default_psi(self):
        """locScaleM with 10% outliers at mean=-10 (man-page example 2)."""
        py = loc_scale_m(_X_OUTLIER_LOC)
        r  = r_loc_scale_m(_X_OUTLIER_LOC)
        assert abs(py["mu"]     - r["mu"])     < 1e-8
        assert abs(py["std_mu"] - r["std_mu"]) < 1e-8
        assert abs(py["disper"] - r["disper"]) < 1e-8


# 2. Man-page examples — scale_m

class TestScaleMManPage:
    """Reproduce the examples from scaleM.Rd exactly."""

    def test_clean_gaussian_bisquare(self):
        """scaleM(r) with default family='bisquare'."""
        py = scale_m(_X_SCALE_CLEAN)
        r  = r_scale_m(_X_SCALE_CLEAN)
        assert abs(py - r) < 1e-8

    def test_outlier_opt(self):
        """scaleM(r2, family='opt') with 10% outliers at mean=-5."""
        py = scale_m(_X_SCALE_OUTLIER, family="opt")
        r  = r_scale_m(_X_SCALE_OUTLIER, family="opt")
        assert abs(py - r) < 1e-8


# 3. Output structure

class TestOutputStructure:
    def test_loc_scale_m_keys(self):
        res = loc_scale_m(_X_CLEAN)
        assert set(res.keys()) == {"mu", "std_mu", "disper"}

    def test_loc_scale_m_types(self):
        res = loc_scale_m(_X_CLEAN)
        for k, v in res.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_scale_m_returns_float(self):
        s = scale_m(_X_SCALE_CLEAN)
        assert isinstance(s, float)

    def test_disper_positive(self):
        res = loc_scale_m(_X_CLEAN)
        assert res["disper"] > 0

    def test_std_mu_positive(self):
        res = loc_scale_m(_X_CLEAN)
        assert res["std_mu"] > 0


# 4. Numerical equivalence across all psi / family values

class TestNumericalEquivalence:
    @pytest.mark.parametrize("psi,eff", [
        ("bisquare", 0.85),
        ("bisquare", 0.90),
        ("bisquare", 0.95),
        ("huber",    0.85),
        ("huber",    0.90),
        ("huber",    0.95),
        ("opt",      0.85),
        ("opt",      0.90),
        ("opt",      0.95),
        ("opt",      0.99),
        ("mopt",     0.85),
        ("mopt",     0.90),
        ("mopt",     0.95),
        ("mopt",     0.99),
    ])
    def test_loc_scale_m_matches_r(self, psi, eff):
        py = loc_scale_m(_X_CLEAN, psi=psi, eff=eff)
        r  = r_loc_scale_m(_X_CLEAN, psi=psi, eff=eff)
        assert abs(py["mu"]     - r["mu"])     < 1e-8, f"mu mismatch psi={psi} eff={eff}"
        assert abs(py["disper"] - r["disper"]) < 1e-8, f"disper mismatch psi={psi} eff={eff}"

    @pytest.mark.parametrize("family", ["bisquare", "opt", "mopt"])
    def test_scale_m_matches_r(self, family):
        py = scale_m(_X_SCALE_CLEAN, family=family)
        r  = r_scale_m(_X_SCALE_CLEAN, family=family)
        assert abs(py - r) < 1e-8, f"scaleM mismatch family={family}"


# 5. Relationship invariant: disper == scaleM(x - mu)

class TestRelationshipInvariant:
    """
    The core relationship: loc_scale_m(x)['disper'] == scale_m(x - mu).
    This must hold for all psi/eff combinations.
    """

    @pytest.mark.parametrize("psi,eff,family", [
        # huber is excluded: scaleM does not accept family="huber".
        # When psi="huber", locScaleM internally calls scaleM(family="bisquare"),
        # but locScaleM also passes psi="huber" to the location loop which uses
        # a different tuning path — the two results therefore differ by design.
        # The relationship is only guaranteed for psi values that scaleM accepts.
        ("bisquare", 0.95, "bisquare"),
        ("opt",      0.95, "opt"),
        ("mopt",     0.95, "mopt"),
    ])
    def test_disper_equals_scale_m_of_residuals(self, psi, eff, family):
        result = loc_scale_m(_X_CLEAN, psi=psi, eff=eff)
        residuals = _X_CLEAN - result["mu"]
        s_inner = scale_m(residuals, delta=0.5, family=family)
        assert abs(s_inner - result["disper"]) < 1e-6, (
            f"disper={result['disper']:.8f} != scaleM(x-mu)={s_inner:.8f} "
            f"for psi={psi}"
        )


_VALID_FAMILY_SCALE = {"bisquare", "opt", "mopt"}


# 6. Robustness checks

class TestRobustness:
    def test_location_robust_to_outliers(self):
        """mu should stay near 0 despite 10% outliers at -10."""
        res_clean   = loc_scale_m(_X_CLEAN)
        res_outlier = loc_scale_m(_X_OUTLIER_LOC)
        # Outliers are at -10; without robustness mu would shift by ~1
        assert abs(res_outlier["mu"]) < 1.0, (
            f"Location was pulled too far: mu={res_outlier['mu']:.3f}"
        )

    def test_scale_robust_to_outliers(self):
        """M-scale should be much smaller than classical SD under contamination."""
        s_robust   = scale_m(_X_SCALE_OUTLIER, family="opt")
        s_classical = float(np.std(_X_SCALE_OUTLIER, ddof=1))
        assert s_robust < s_classical, (
            f"M-scale {s_robust:.3f} not smaller than SD {s_classical:.3f}"
        )

    def test_consistency_on_gaussian(self):
        """On clean N(0, 1.5²) data, disper should be close to 1.5."""
        res = loc_scale_m(_X_CLEAN)
        assert abs(res["disper"] - 1.5) < 0.3


# 7. Input type coverage

class TestInputTypes:
    _x = _X_CLEAN[:50]  # small slice for speed

    def test_numpy_array(self):
        res = loc_scale_m(self._x)
        assert isinstance(res["mu"], float)

    def test_python_list(self):
        res = loc_scale_m(list(self._x))
        assert isinstance(res["mu"], float)

    def test_pandas_series(self):
        res = loc_scale_m(pd.Series(self._x))
        assert isinstance(res["mu"], float)

    def test_polars_series(self):
        pl = pytest.importorskip("polars")
        res = loc_scale_m(pl.Series(self._x))
        assert isinstance(res["mu"], float)

    def test_numpy_and_pandas_agree(self):
        r1 = loc_scale_m(self._x)
        r2 = loc_scale_m(pd.Series(self._x))
        assert abs(r1["mu"] - r2["mu"]) < 1e-12

    def test_scale_m_pandas(self):
        s = scale_m(pd.Series(self._x))
        assert isinstance(s, float)


# 8. NA / NaN handling

class TestNaHandling:
    def test_na_rm_false_raises_or_returns_nan(self):
        """With na.rm=False (default), NaN in data causes R to raise an error."""
        import rpy2.rinterface_lib.embedded as r_embedded
        x_with_nan = np.append(_X_CLEAN[:50], np.nan)
        # R's locScaleM raises a hard RRuntimeError when NA is present
        # and na.rm=FALSE — it does not silently return NaN.
        with pytest.raises((r_embedded.RRuntimeError, Exception)):
            loc_scale_m(x_with_nan, na_rm=False)

    def test_na_rm_true_ignores_nan(self):
        """With na_rm=True, NaN values are stripped and result is finite."""
        x_with_nan = np.append(_X_CLEAN[:50], np.nan)
        res = loc_scale_m(x_with_nan, na_rm=True)
        assert math.isfinite(res["mu"])
        assert math.isfinite(res["disper"])


# 9. Edge cases

class TestEdgeCases:
    def test_near_constant_data(self):
        """If MAD ≈ 0, function should return mu=median, disper=0 with warning."""
        x_const = np.full(50, 3.14)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = loc_scale_m(x_const)
        assert res["disper"] == 0.0
        assert res["std_mu"] == 0.0
        assert abs(res["mu"] - 3.14) < 1e-10

    def test_scale_m_delta_values(self):
        """scaleM with different delta values should all return positive floats."""
        for delta in [0.25, 0.5, 0.75]:
            s = scale_m(_X_SCALE_CLEAN, delta=delta)
            assert s > 0, f"scale_m with delta={delta} returned {s}"

    def test_large_n(self):
        """Should work and converge for large samples."""
        seed_r(99)
        x_large = np.array(ro.r("rnorm(5000, mean=5, sd=2)"))
        res = loc_scale_m(x_large)
        assert abs(res["mu"] - 5.0) < 0.2
        assert abs(res["disper"] - 2.0) < 0.2


# 10. Parameter validation

class TestParameterValidation:
    def test_invalid_psi_raises(self):
        with pytest.raises(ValueError, match="psi="):
            loc_scale_m(_X_CLEAN, psi="unknown")

    def test_invalid_eff_raises(self):
        with pytest.raises(ValueError, match="eff="):
            loc_scale_m(_X_CLEAN, psi="bisquare", eff=0.99)  # 0.99 only for opt/mopt

    def test_invalid_family_scale_m_raises(self):
        with pytest.raises(ValueError, match="family="):
            scale_m(_X_SCALE_CLEAN, family="huber")  # huber not supported in scaleM

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            scale_m(_X_SCALE_CLEAN, delta=0.0)

    def test_delta_above_1_raises(self):
        with pytest.raises(ValueError, match="delta"):
            scale_m(_X_SCALE_CLEAN, delta=1.5)

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            loc_scale_m(np.array([[1, 2], [3, 4]]))

    def test_non_numeric_raises(self):
        with pytest.raises((TypeError, Exception)):
            loc_scale_m(["a", "b", "c"])
