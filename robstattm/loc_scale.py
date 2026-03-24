"""
loc_scale.py — Python wrappers for RobStatTM::locScaleM and RobStatTM::scaleM.

Both wrappers delegate all computation to the R package via RPy2.
They accept numpy arrays, pandas Series, and polars Series as input,
and return plain Python dicts / floats.

Notes on the relationship between the two functions
----------------------------------------------------
``loc_scale_m`` implements Section 2.7.1 of Maronna et al. (2019):

  1. Initialise  mu_0 = median(x),  sigma_0 = MAD(x)  (MADN = MAD/0.675).
  2. Run an IRLS loop that updates **only mu**, keeping sigma_0 fixed
     as a plug-in dispersion estimate.
  3. After the location loop converges, call ``scaleM`` once on the
     **centred residuals** x - mu_final to obtain the final dispersion.

``scale_m`` is therefore the *inner scale step*, exposed as a standalone
function. It solves the M-scale equation:

    mean( rho(u / s, cc) ) = delta

iteratively via  s^2 <- s^2 * mean(rho(u/s, cc)) / delta,
starting from the scaled median of |u|.

Consequently:

    loc_scale_m(x)["disper"]  ==  scale_m(x - loc_scale_m(x)["mu"])

up to floating-point precision, provided the same ``psi``/``family``
and ``delta`` are used.
"""

from ._utils import _rpy2, _to_r_vector, _from_r_scalar, _from_r_list

# Valid parameter values (mirrored from R man pages)
_VALID_PSI = {"bisquare", "huber", "opt", "mopt"}
_VALID_EFF = {0.85, 0.90, 0.95}
_VALID_EFF_OPT = {0.85, 0.90, 0.95, 0.99}
_VALID_FAMILY_SCALE = {"bisquare", "opt", "mopt"}


def _get_robstattm():
    """Return the imported RobStatTM R package (cached after first call)."""
    ro = _rpy2()
    try:
        from rpy2.robjects.packages import importr
        return importr("RobStatTM")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the RobStatTM R package. "
            "Make sure it is installed in R:  install.packages('RobStatTM')"
        ) from exc


# loc_scale_m

def loc_scale_m(x, psi="mopt", eff=0.95, maxit=50, tol=1e-4, na_rm=False):
    """
    Robust joint M-estimator of location and scale (univariate).

    Computes simultaneous M-estimates of location (``mu``) and scale
    (``disper``) for a univariate sample using an IRLS (Iteratively
    Re-Weighted Least Squares) algorithm, together with an estimated
    standard deviation of the location estimate (``std_mu``).

    The location loop is initialised with the sample median and MAD, and
    uses a fixed plug-in dispersion (Section 2.7.1 of Maronna et al., 2019).
    After the location estimate converges, the M-scale is computed via
    :func:`scale_m` applied to the centred residuals.

    This is a Python wrapper around ``RobStatTM::locScaleM`` (R) via RPy2.
    All numerical computation is performed in R.

    Parameters
    ----------
    x : array-like
        1-D numeric vector of observations. Accepts ``numpy.ndarray``,
        ``pandas.Series``, ``polars.Series``, or any Python sequence
        convertible to a float array.
    psi : {"mopt", "bisquare", "huber", "opt"}, optional
        Name of the psi (score) function. Default is ``"mopt"``.

        - ``"bisquare"`` — Tukey bisquare (redescending). Available
          efficiencies: 0.85, 0.90, 0.95.
        - ``"huber"``    — Huber (monotone). Available efficiencies:
          0.85, 0.90, 0.95.
        - ``"opt"``      — Optimal bias-robust. Available efficiencies:
          0.85, 0.90, 0.95, 0.99.
        - ``"mopt"``     — Modified optimal (default). Available
          efficiencies: 0.85, 0.90, 0.95, 0.99.

    eff : float, optional
        Desired asymptotic efficiency relative to the MLE under a Gaussian
        model. Default is ``0.95``. Valid values depend on *psi*; see above.
    maxit : int, optional
        Maximum number of IRLS iterations for the location loop.
        Default is ``50``.
    tol : float, optional
        Convergence tolerance. The loop stops when
        ``|mu_new - mu_old| / sigma_0 < tol``. Default is ``1e-4``.
    na_rm : bool, optional
        If ``True``, ``NaN`` / ``NA`` values are silently removed before
        computation. Default is ``False``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``"mu"`` : float
            Robust M-estimate of location.
        ``"std_mu"`` : float
            Estimated standard deviation of ``mu``, derived from the
            asymptotic variance formula (eq. 2.65, Maronna et al., 2019)
            with sample-average plug-ins.
        ``"disper"`` : float
            M-scale (dispersion) estimate, obtained by calling
            ``scaleM(x - mu, delta=0.5, family=psi)`` after the location
            loop converges.

    Raises
    ------
    ImportError
        If ``rpy2`` is not installed.
    RuntimeError
        If the ``RobStatTM`` R package is not installed.
    ValueError
        If *x* is not 1-dimensional, or *psi* / *eff* values are invalid.
    TypeError
        If *x* cannot be coerced to a numeric array.

    See Also
    --------
    scale_m : M-scale estimator with a fixed location parameter.

    Notes
    -----
    The tuning constant *k* that determines the ψ cutoff is looked up from
    a pre-computed table inside the R package, keyed on ``(psi, eff)``.
    For ``psi="bisquare"`` and ``eff=0.95`` this constant is 4.685
    (Table 2.4, Maronna et al., 2019).

    If ``MAD(x) < 1e-10`` (data effectively constant), the function
    returns ``mu = median(x), std_mu = 0, disper = 0`` and emits a
    warning about repeated values.

    References
    ----------
    Maronna, R. A., Martin, R. D., Yohai, V. J., and Salibian-Barrera, M.
    (2019). *Robust Statistics: Theory and Methods (with R)*, 2nd ed.
    Wiley. https://www.wiley.com/go/maronna/robust

    Examples
    --------
    Basic usage on clean Gaussian data:

    >>> import numpy as np
    >>> from robstattm import loc_scale_m
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(loc=2.0, scale=1.5, size=150)
    >>> result = loc_scale_m(x)
    >>> abs(result["mu"] - 2.0) < 0.3        # near true location
    True
    >>> abs(result["disper"] - 1.5) < 0.3    # near true scale
    True

    Robustness under 10 % contamination:

    >>> rng = np.random.default_rng(0)
    >>> good = rng.normal(loc=0, scale=1.5, size=135)
    >>> bad  = rng.normal(loc=-10, scale=0.5, size=15)
    >>> x2   = np.concatenate([good, bad])
    >>> result2 = loc_scale_m(x2)
    >>> abs(result2["mu"]) < 0.5             # still near 0 despite outliers
    True

    Using a different psi function and efficiency:

    >>> result3 = loc_scale_m(x, psi="bisquare", eff=0.90)

    Reproducing the exact R man-page example (requires set.seed via R):

    >>> import rpy2.robjects as ro                       # doctest: +SKIP
    >>> ro.r("set.seed(123)")                            # doctest: +SKIP
    >>> x_r = np.array(ro.r("rnorm(150, sd=1.5)"))      # doctest: +SKIP
    >>> py_result = loc_scale_m(x_r)                    # doctest: +SKIP
    >>> r_result  = ro.r("locScaleM(x_r)")              # doctest: +SKIP
    """
    # input validation 
    if psi not in _VALID_PSI:
        raise ValueError(
            f"psi={psi!r} is not valid. Choose from {sorted(_VALID_PSI)}."
        )
    allowed_eff = _VALID_EFF_OPT if psi in {"opt", "mopt"} else _VALID_EFF
    if eff not in allowed_eff:
        raise ValueError(
            f"eff={eff} is not valid for psi={psi!r}. "
            f"Choose from {sorted(allowed_eff)}."
        )

    # convert to R vector
    x_r = _to_r_vector(x)

    # call R function
    ro = _rpy2()
    pkg = _get_robstattm()

    r_result = pkg.locScaleM(
        x_r,
        psi=psi,
        eff=ro.FloatVector([eff]),
        maxit=ro.IntVector([maxit]),
        tol=ro.FloatVector([tol]),
        **{"na.rm": ro.BoolVector([na_rm])},
    )

    # extract results
    mu, std_mu, disper = _from_r_list(r_result, "mu", "std.mu", "disper")

    return {"mu": mu, "std_mu": std_mu, "disper": disper}


# scale_m

def scale_m(
    u,
    delta=0.5,
    family="bisquare",
    max_it=100,
    tol=1e-6,
    tuning_chi=None,
):
    """
    Robust M-scale estimator (univariate).

    Computes a robust estimate of spread (scale) by solving the implicit
    M-scale equation:

        mean( rho(u / s, cc) ) = delta

    where ``rho`` is a bounded loss function parameterised by a tuning
    constant ``cc`` chosen to make the estimator consistent for the standard
    deviation under a Gaussian model.

    The iterative algorithm (Section 2.8.2, Maronna et al., 2019) starts
    from the scaled median absolute value and updates via:

        s² ← s² * mean( rho(u/s, cc) ) / delta

    until ``|s_new/s_old - 1| < tol``.

    This is a Python wrapper around ``RobStatTM::scaleM`` (R) via RPy2.
    All numerical computation is performed in R.

    Parameters
    ----------
    u : array-like
        1-D numeric vector of residuals (typically ``x - mu`` where ``mu``
        is a location estimate). Accepts ``numpy.ndarray``, ``pandas.Series``,
        ``polars.Series``, or any Python sequence convertible to float.
    delta : float, optional
        Right-hand side of the M-scale equation. Controls the breakdown
        point: BP = min(delta, 1 - delta). Default is ``0.5``, which
        maximises the breakdown point.
    family : {"bisquare", "opt", "mopt"}, optional
        Name of the rho (loss) function. Default is ``"bisquare"``.
    max_it : int, optional
        Maximum number of iterations. Default is ``100``.
    tol : float, optional
        Relative convergence tolerance: stop when
        ``|s_new / s_old - 1| < tol``. Default is ``1e-6``.
    tuning_chi : float or None, optional
        Override the tuning constant ``cc``. If ``None`` (default), the
        constant is computed by the R function ``lmrobdet.control`` to
        ensure Fisher consistency at the given ``family`` and ``delta``.
        Only pass this if you know what you are doing.

    Returns
    -------
    float
        The M-scale estimate at convergence (or at the last iteration if
        the algorithm did not converge within ``max_it`` steps).

    Raises
    ------
    ImportError
        If ``rpy2`` is not installed.
    RuntimeError
        If the ``RobStatTM`` R package is not installed.
    ValueError
        If ``family`` is not one of the valid options, or if ``delta`` is
        not in (0, 1).
    TypeError
        If ``u`` cannot be coerced to a numeric array.

    See Also
    --------
    loc_scale_m : Joint M-estimator of location and scale.

    Notes
    -----
    ``scale_m`` is the inner scale-update loop of ``loc_scale_m`` exposed
    as a standalone function. Specifically:

        ``loc_scale_m(x)["disper"]``  ==  ``scale_m(x - loc_scale_m(x)["mu"])``

    up to floating-point precision, provided the same ``family``/``delta``
    are used. The difference is that ``loc_scale_m`` updates the location
    simultaneously (using a fixed MAD as the working scale), then calls
    ``scale_m`` once on the centred data; ``scale_m`` itself only iterates
    the scale equation, treating location as fixed.

    If all residuals are identically zero (``max(|u|) < tolerancezero``),
    the function returns ``0.0``.

    The breakdown point of the estimator is ``min(delta, 1-delta)``, so the
    default ``delta=0.5`` gives the maximum possible breakdown point of 50 %.

    References
    ----------
    Maronna, R. A., Martin, R. D., Yohai, V. J., and Salibian-Barrera, M.
    (2019). *Robust Statistics: Theory and Methods (with R)*, 2nd ed.
    Wiley. https://www.wiley.com/go/maronna/robust

    Examples
    --------
    Scale estimate on clean data (should be close to true sd=1.5):

    >>> import numpy as np
    >>> from robstattm import scale_m
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(scale=1.5, size=150)
    >>> s = scale_m(x)
    >>> abs(s - 1.5) < 0.3
    True

    Robustness under 10 % contamination:

    >>> rng = np.random.default_rng(0)
    >>> good = rng.normal(scale=1.5, size=135)
    >>> bad  = rng.normal(loc=-5, scale=0.5, size=15)
    >>> x2   = np.concatenate([good, bad])
    >>> s2 = scale_m(x2, family="opt")
    >>> s_classical = float(np.std(x2, ddof=1))
    >>> s2 < s_classical   # M-scale is much less inflated
    True

    Relationship to loc_scale_m:

    >>> from robstattm import loc_scale_m
    >>> result = loc_scale_m(x)
    >>> s_inner = scale_m(x - result["mu"])
    >>> abs(s_inner - result["disper"]) < 1e-6   # must be equal
    True

    Reproducing the R man-page example:

    >>> import rpy2.robjects as ro                    # doctest: +SKIP
    >>> ro.r("set.seed(123)")                         # doctest: +SKIP
    >>> x_r = np.array(ro.r("rnorm(150, sd=1.5)"))   # doctest: +SKIP
    >>> scale_m(x_r)                                 # doctest: +SKIP
    """
    # input validation
    if family not in _VALID_FAMILY_SCALE:
        raise ValueError(
            f"family={family!r} is not valid for scale_m. "
            f"Choose from {sorted(_VALID_FAMILY_SCALE)}."
        )
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1); got {delta}.")

    # convert to R vector
    u_r = _to_r_vector(u)

    # call R function
    ro = _rpy2()
    pkg = _get_robstattm()

    kwargs = dict(
        delta=ro.FloatVector([delta]),
        family=ro.StrVector([family]),
        **{"max.it": ro.IntVector([max_it])},
        tol=ro.FloatVector([tol]),
    )
    # Only pass tuning.chi if the caller explicitly provided one
    if tuning_chi is not None:
        kwargs["tuning.chi"] = ro.FloatVector([float(tuning_chi)])

    r_result = pkg.scaleM(u_r, **kwargs)

    return _from_r_scalar(r_result)
