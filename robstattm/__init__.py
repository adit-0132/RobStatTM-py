"""
robstattm: Python wrappers for the RobStatTM R package.

Provides Python interfaces to robust estimation functions from the
RobStatTM R package (Maronna, Martin, Yohai, Salibian-Barrera, 2019).
Input accepts numpy arrays, pandas Series, and polars Series.
"""

from .loc_scale import loc_scale_m, scale_m

__all__ = ["loc_scale_m", "scale_m"]
__version__ = "0.1.0"
