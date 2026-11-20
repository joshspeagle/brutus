"""
Mathematical, photometric, and statistical sampling utilities.
"""

from .math import (
    chisquare_logpdf,
    galactic_to_galactocentric,
    galactic_to_galactocentric_cyl,
    inverse3,
    isPSD,
    pm_equatorial_to_galactic,
    truncnorm_logpdf,
    truncnorm_pdf,
)
from .photometry import (
    add_mag,
    chisquare_outlier_loglike,
    inv_luptitude,
    inv_magnitude,
    luptitude,
    magnitude,
    phot_loglike,
    uniform_outlier_loglike,
)
from .sampling import draw_sar, quantile, sample_multivariate_normal

__all__ = [
    "magnitude",
    "inv_magnitude",
    "luptitude",
    "inv_luptitude",
    "add_mag",
    "phot_loglike",
    "chisquare_outlier_loglike",
    "uniform_outlier_loglike",
    "galactic_to_galactocentric",
    "galactic_to_galactocentric_cyl",
    "pm_equatorial_to_galactic",
    "inverse3",
    "isPSD",
    "chisquare_logpdf",
    "truncnorm_pdf",
    "truncnorm_logpdf",
    "quantile",
    "sample_multivariate_normal",
    "draw_sar",
]
