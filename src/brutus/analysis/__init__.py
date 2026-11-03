"""
Fit individual stars, coeval star clusters, 3D line-of-sight dust, and offsets.

Provides grid-based Bayesian inference for individual stars (`BruteForce`),
isochrone mixture-model likelihoods for stellar populations
(`isochrone_population_loglike`), multi-cloud line-of-sight dust extinction
models (`los_clouds_loglike_samples`, `los_clouds_priortransform`), and
leave-one-out photometric offset calibration (`photometric_offsets`).
"""

from .individual import BruteForce
from .los_dust import (
    kernel_gauss,
    kernel_lorentz,
    kernel_tophat,
    los_clouds_loglike_samples,
    los_clouds_priortransform,
)
from .offsets import PhotometricOffsetsConfig, photometric_offsets
from .populations import (
    apply_isochrone_mixture_model,
    compute_isochrone_cluster_loglike,
    compute_isochrone_outlier_loglike,
    generate_isochrone_population_grid,
    isochrone_population_loglike,
    marginalize_isochrone_grid,
)

__all__ = [
    # Individual star fitting
    "BruteForce",
    # Stellar population analysis (isochrone-based)
    "isochrone_population_loglike",
    "generate_isochrone_population_grid",
    "compute_isochrone_cluster_loglike",
    "compute_isochrone_outlier_loglike",
    "apply_isochrone_mixture_model",
    "marginalize_isochrone_grid",
    # Photometric offset analysis
    "photometric_offsets",
    "PhotometricOffsetsConfig",
    # Line-of-sight dust extinction analysis
    "los_clouds_priortransform",
    "los_clouds_loglike_samples",
    "kernel_tophat",
    "kernel_gauss",
    "kernel_lorentz",
]
