"""
_utils.py — internal type conversion helpers.

All public functions in this package call _to_r_vector() to normalise
Python inputs before handing them to RPy2, and _from_r_scalar() /
_from_r_list() to extract results from R ListVectors.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Lazy RPy2 imports — only materialised when a conversion is actually needed.
# This lets the module be imported for documentation / type-checking purposes
# even on machines where RPy2 is not installed.
# ---------------------------------------------------------------------------

def _rpy2():
    """Return the rpy2.robjects module, raising a clear error if absent."""
    try:
        import rpy2.robjects as ro
        return ro
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to call RobStatTM functions. "
            "Install it with:  pip install rpy2"
        ) from exc


def _to_r_vector(x):
    """
    Convert a Python array-like to an R numeric vector (FloatVector).

    Parameters
    ----------
    x : array-like
        Accepts numpy.ndarray, pandas.Series, polars.Series, or any
        Python sequence that can be coerced to a 1-D float array.

    Returns
    -------
    rpy2.robjects.vectors.FloatVector
        A 1-D R numeric vector containing the values of *x*.

    Raises
    ------
    TypeError
        If *x* cannot be coerced to a numeric 1-D array.
    ValueError
        If *x* is not 1-dimensional after coercion.
    """
    ro = _rpy2()

    # polars Series → numpy
    try:
        import polars as pl
        if isinstance(x, pl.Series):
            x = x.to_numpy()
    except ImportError:
        pass

    # pandas Series → numpy
    try:
        import pandas as pd
        if isinstance(x, pd.Series):
            x = x.to_numpy()
    except ImportError:
        pass

    # everything else → numpy float64
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(
            f"Input must be 1-dimensional; got shape {arr.shape}."
        )

    return ro.FloatVector(arr)


def _from_r_scalar(r_obj):
    """Extract a Python float from a length-1 R vector."""
    return float(r_obj[0])


def _from_r_list(r_list, *keys):
    """
    Extract named elements from an R list (ListVector).

    Parameters
    ----------
    r_list : rpy2 ListVector
        The R list returned by a function such as locScaleM.
    *keys : str
        Names of the elements to extract, in order.

    Returns
    -------
    tuple of float
        One float per key, in the same order as *keys*.
    """
    return tuple(float(r_list.rx2(k)[0]) for k in keys)
