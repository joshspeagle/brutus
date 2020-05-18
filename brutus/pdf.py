#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF functions.

"""

from __future__ import (print_function, division)

import sys
import os
import warnings
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates import CylindricalRepresentation as CylRep
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter as norm_kde
import copy

from .utils import draw_sar, _truncnorm_logpdf
from .dust import Bayestar

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["imf_lnprior", "ps1_MrLF_lnprior", "parallax_lnprior",
           "scale_parallax_lnprior", "parallax_to_scale",
           "logn_disk", "logn_halo",
           "logp_feh", "logp_age_from_feh",
           "gal_lnprior", "dust_lnprior",
           "bin_pdfs_distred"]


def imf_lnprior(mgrid, alpha_low=1.3, alpha_high=2.3, mass_break=0.5,
                mgrid2=None):
    """
    Apply a Kroupa-like broken IMF prior over the provided initial mass grid.

    Parameters
    ----------
    mgrid : `~numpy.ndarray` of shape (Ngrid)
        Grid of initial mass (solar units) the IMF will be evaluated over.

    alpha_low : float, optional
        Power-law slope for the low-mass component of the IMF.
        Default is `1.3`.

    alpha_high : float, optional
        Power-law slope for the high-mass component of the IMF.
        Default is `2.3`.

    mass_break : float, optional
        The mass where we transition from `alpha_low` to `alpha_high`.
        Default is `0.5`.

    mgrid2 : `~numpy.ndarray` of shape (Ngrid)
        Grid of initial mass (solar units) for the second portion of a binary
        that the IMF will be evaluated over.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (Ngrid)
        The corresponding unnormalized ln(prior).

    """

    # Initialize log-prior.
    lnprior = np.zeros_like(mgrid) - np.inf

    # Low mass.
    low_mass = (mgrid <= mass_break) & (mgrid > 0.08)
    lnprior[low_mass] = -alpha_low * np.log(mgrid[low_mass])

    # High mass.
    high_mass = mgrid > mass_break
    lnprior[high_mass] = (-alpha_high * np.log(mgrid[high_mass])
                          + (alpha_high - alpha_low) * np.log(mass_break))

    # Compute normalization.
    norm_low = mass_break ** (1. - alpha_low) / (alpha_high - 1.)
    norm_high = 0.08 ** (1. - alpha_low) / (alpha_low - 1.)  # H-burning limit
    norm_high -= mass_break ** (1. - alpha_low) / (alpha_low - 1.)
    norm = norm_low + norm_high

    # Compute contribution from binary component.
    if mgrid2 is not None:
        lnprior2 = np.zeros_like(mgrid2) - np.inf

        # Low mass.
        low_mass = (mgrid2 <= mass_break) & (mgrid2 > 0.08)
        lnprior2[low_mass] = -alpha_low * np.log(mgrid2[low_mass])

        # High mass.
        high_mass = mgrid2 > mass_break
        lnprior2[high_mass] = (-alpha_high * np.log(mgrid2[high_mass])
                               + (alpha_high - alpha_low) * np.log(mass_break))

        # Combine with primary.
        lnprior += lnprior2

        # Compute new normalization.
        norm = (norm_low ** 2 + norm_high ** 2 + 2 * norm_low * norm_high)

    return lnprior - np.log(norm)


def ps1_MrLF_lnprior(Mr):
    """
    Apply PanSTARRS r-band luminosity function-based prior over the provided
    absolute r-band magnitude grid.

    Parameters
    ----------
    Mr : `~numpy.ndarray` of shape (Ngrid)
        Grid of PS1 absolute r-band magnitudes.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (Ngrid)
        The corresponding unnormalized ln(prior).

    """

    global ps_lnprior
    try:
        # Evaluate prior from file.
        lnprior = ps_lnprior(Mr)
    except:
        # Read in file.
        path = os.path.dirname(os.path.realpath(__file__))
        grid_Mr, grid_lnp = np.loadtxt(path + '/PSMrLF_lnprior.dat').T
        # Construct prior.
        ps_lnprior = interp1d(grid_Mr, grid_lnp, fill_value='extrapolate')
        # Evaluate prior.
        lnprior = ps_lnprior(Mr)

    return lnprior


def parallax_lnprior(parallaxes, p_meas, p_err):
    """
    Apply parallax prior using a measured parallax.

    Parameters
    ----------
    parallaxes : `~numpy.ndarray` of shape (N)
        Parallaxes.

    p_meas : float
        Measured parallax.

    p_std : float
        Measured parallax error.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (N)
        The corresponding ln(prior).

    """

    if np.isfinite(p_meas) and np.isfinite(p_err):
        # Compute log-prior.
        chi2 = (parallaxes - p_meas)**2 / p_err**2  # chi2
        lnorm = np.log(2. * np.pi * p_err**2)  # normalization
        lnprior = -0.5 * (chi2 + lnorm)
    else:
        # If no measurement, assume a uniform prior everywhere.
        lnprior = np.zeros_like(parallaxes)

    return lnprior


def scale_parallax_lnprior(scales, scale_errs, p_meas, p_err, snr_lim=4.):
    """
    Apply parallax prior to a set of flux density scalefactors
    `s ~ p**2` using a measured parallax.

    Parameters
    ----------
    scales : `~numpy.ndarray` of shape (N)
        Scale-factors (`s = p**2`).

    scale_errs : `~numpy.ndarray` of shape (N)
        Scale-factor errors.

    p_meas : float
        Measured parallax.

    p_std : float
        Measured parallax error.

    snr_lim : float, optional
        The signal-to-noise ratio limit used to apply the approximation.
        If `snr < snr_lim`, then a uniform prior will be returned instead.
        Default is `4.`.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (N)
        The corresponding ln(prior).

    """

    if np.isfinite(p_meas) and np.isfinite(p_err) and p_meas / p_err > snr_lim:
        # Convert from `p` to `s=p**2` space assuming roughly Normal.
        s_mean, s_std = parallax_to_scale(p_meas, p_err)

        # Compute log-prior.
        svar_tot = s_std**2 + scale_errs**2
        chi2 = (scales - s_mean)**2 / svar_tot  # chi2
        lnorm = np.log(2. * np.pi * svar_tot)  # normalization
        lnprior = -0.5 * (chi2 + lnorm)
    else:
        # If no measurement, assume a uniform prior everywhere.
        lnprior = np.zeros_like(scales)

    return lnprior


def parallax_to_scale(p_meas, p_err, snr_lim=4.):
    """
    Convert parallax flux density scalefactor `s ~ p**2`.

    Parameters
    ----------
    p_meas : float
        Measured parallax.

    p_std : float
        Measured parallax error.

    snr_lim : float, optional
        The signal-to-noise ratio limit used to apply the approximation.
        If `snr < snr_lim`, then `s_std = 1e20` will be returned.
        Default is `4.`.

    Returns
    -------
    s_mean : float
        Corresponding mean of the scale-factor.

    s_std : float
        Corresponding standard deviation of the scale-factor.

    """

    if p_meas / p_err > snr_lim:
        # Convert from `p` to `s=p**2` space assuming roughly Normal.
        pm, pe = max(0., p_meas), p_err  # floor to 0
        s_mean = pm**2 + pe**2  # scale mean
        s_std = np.sqrt(2 * pe**4 + 4 * pm**2 * pe**2)  # scale stddev
    else:
        s_mean, s_std = 1e-20, 1e20

    return s_mean, s_std


def logn_disk(R, Z, R_solar=8.2, Z_solar=0.025, R_scale=2.6, Z_scale=0.3):
    """
    Log-number density of stars in the disk component of the galaxy.

    Parameters
    ----------
    R : `~numpy.ndarray` of shape (N)
        The distance from the center of the galaxy.

    Z : `~numpy.ndarray` of shape (N)
        The height above the galactic midplane.

    R_solar : float, optional
        The solar distance from the center of the galaxy in kpc.
        Default is `8.2`.

    Z_solar : float, optional
        The solar height above the galactic midplane in kpc.
        Default is `0.025`.

    R_scale : float, optional
        The scale radius of the disk in kpc. Default is `2.6`.

    Z_scale : float, optional
        The scale height of the disk in kpc. Default is `0.3`.

    Returns
    -------
    logn : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(number density).

    """

    rterm = (R - R_solar) / R_scale  # radius term
    zterm = (np.abs(Z) - np.abs(Z_solar)) / Z_scale  # height term

    return -(rterm + zterm)


def logn_halo(R, Z, R_solar=8.2, Z_solar=0.025, R_smooth=1.0,
              eta=4.2, q_ctr=0.2, q_inf=0.8, r_q=6.):
    """
    Log-number density of stars in the halo component of the galaxy.

    Parameters
    ----------
    R : `~numpy.ndarray` of shape (N)
        The distance from the center of the galaxy.

    Z : `~numpy.ndarray` of shape (N)
        The height above the galactic midplane.

    R_solar : float, optional
        The solar distance from the center of the galaxy in kpc.
        Default is `8.2`.

    Z_solar : float, optional
        The solar height above the galactic midplane in kpc.
        Default is `0.025`.

    R_smooth : float, optional
        The smoothing radius in kpc used to avoid singularities
        around the Galactic center. Default is `1.0`.

    eta : float, optional
        The (negative) power law index describing the number density.
        Default is `4.2`.

    q_ctr : float, optional
        The nominal oblateness of the halo at Galactic center.
        Default is `0.2`.

    q_inf : float, optional
        The nominal oblateness of the halo infinitely far away.
        Default is `0.8`.

    r_q : float, optional
        The scale radius over which the oblateness changes in kpc.
        Default is `6.`.

    Returns
    -------
    logn : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(number density).

    """

    # Compute distance from Galactic center.
    r = np.sqrt(R**2 + Z**2)

    # Compute oblateness.
    rp = np.sqrt(r**2 + r_q**2)
    q = q_inf - (q_inf - q_ctr) * np.exp(1. - rp / r_q)

    # Compute effective radius.
    Reff = np.sqrt(R**2 + (Z / q)**2 + R_smooth**2)

    # Compute solar value for normalization.
    rp_solar = np.sqrt(R_solar**2 + Z_solar**2 + r_q**2)
    q_solar = q_inf - (q_inf - q_ctr) * np.exp(1. - rp_solar / r_q)
    Reff_solar = np.sqrt(R_solar**2 + (Z_solar / q_solar) + R_smooth**2)

    # Compute inner component.
    logn = -eta * np.log(Reff / Reff_solar)

    return logn


def logp_feh(feh, feh_mean=-0.2, feh_sigma=0.3):
    """
    Log-prior for the metallicity in a given component of the galaxy.

    Parameters
    ----------
    feh : `~numpy.ndarray` of shape (N)
        The metallicities of the corresponding models.

    feh_mean : float, optional
        The mean metallicity. Default is `-0.2`.

    feh_sigma : float, optional
        The standard deviation in the metallicity. Default is `0.3`.

    Returns
    -------
    logp : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(probability).

    """

    # Compute log-probability.
    chi2 = (feh_mean - feh)**2 / feh_sigma**2  # chi2
    lnorm = np.log(2. * np.pi * feh_sigma**2)  # normalization
    lnprior = -0.5 * (chi2 + lnorm)

    return lnprior


def logp_age_from_feh(age, feh_mean=-0.2, max_age=13.8, min_age=0.,
                      feh_age_ctr=-0.5, feh_age_scale=0.5,
                      nsigma_from_max_age=2., max_sigma=4., min_sigma=1.):
    """
    Log-prior for the age in the disk component of the galaxy. Designed to
    follow the disk metallicity prior.

    Parameters
    ----------
    age : `~numpy.ndarray` of shape (N)
        The ages of the corresponding models whose `Z` has been provided.

    feh_mean : float, optional
        The mean metallicity. Default is `-0.2`.

    max_age : float, optional
        The maximum allowed mean age (in Gyr). Default is `13.8`.

    min_age : float, optional
        The minimum allowed mean age (in Gyr). Default is `0.`.

    feh_age_ctr : float, optional
        The mean metallicity where the mean age is halfway between
        `max_age` and `min_age`. Default is `-0.5`.

    feh_age_scale : float, optional
        The exponential scale-length at which the mean age approaches
        `max_age` or `min_age` as it moves to lower or higher mean metallicity,
        respectively. Default is `0.5`.

    nsigma_from_max_age : float, optional
        The number of sigma away the mean age should be from `max_age`
        (i.e. the mean age is `nsigma_from_max_age`-sigma lower
        than `max_age`). Default is `2.`.

    max_sigma : float, optional
        The maximum allowed sigma (in Gyr). Default is `4.`.

    min_sigma : float, optional
        The minimum allowed sigma (in Gyr). Default is `1.`.

    Returns
    -------
    logp : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(probability).

    """

    # Compute mean age.
    age_mean_pred = ((max_age - min_age)
                     / (1. + np.exp((feh_mean - feh_age_ctr) / feh_age_scale))
                     + min_age)

    # Compute age spread.
    age_sigma_pred = (max_age - age_mean_pred) / nsigma_from_max_age
    age_sigma_pred = min(max(age_sigma_pred, min_sigma), max_sigma)  # bound

    # Compute log-probability.
    a = (min_age - age_mean_pred) / age_sigma_pred
    b = (max_age - age_mean_pred) / age_sigma_pred
    lnprior = _truncnorm_logpdf(age, a, b,
                                loc=age_mean_pred, scale=age_sigma_pred)

    return lnprior


def gal_lnprior(dists, coord, labels=None, R_solar=8.2, Z_solar=0.025,
                R_thin=2.6, Z_thin=0.3,
                R_thick=2.0, Z_thick=0.9, f_thick=0.04,
                Rs_halo=1.0, q_halo_ctr=0.2, q_halo_inf=0.8, r_q_halo=6.0,
                eta_halo=4.2, f_halo=0.005,
                feh_thin=-0.2, feh_thin_sigma=0.3,
                feh_thick=-0.7, feh_thick_sigma=0.4,
                feh_halo=-1.6, feh_halo_sigma=0.5,
                max_age=13.8, min_age=0., feh_age_ctr=-0.5, feh_age_scale=0.5,
                nsigma_from_max_age=2., max_sigma=4., min_sigma=1.,
                return_components=False):
    """
    Log-prior for a galactic model containing a thin disk, thick disk, and
    halo. The default behavior imposes a prior based on the total
    number density from all three components. If the metallicity and/or age is
    provided, then an associated galactic metallicity and/or age model
    is also imposed. Partially based on Bland-Hawthorn & Gerhard (2016).

    Parameters
    ----------
    dists : `~numpy.ndarray` of shape `(N,)`
        Distance from the observer in kpc.

    coord : 2-tuple
        The `(l, b)` galaxy coordinates of the object.

    labels : structured `~numpy.ndarray` of shape `(N, Nlabels)`
        Collection of labels associated with the models whose distance
        estimates are provided. Must contain the label `'feh'` to apply
        the metallicity prior.

    R_solar : float, optional
        The solar distance from the center of the galaxy in kpc.
        Default is `8.2`.

    Z_solar : float, optional
        The solar height above the galactic midplane in kpc.
        Default is `0.025`.

    R_thin : float, optional
        The scale radius of the thin disk in kpc. Default is `2.6`.

    Z_thin : float, optional
        The scale height of the thin disk in kpc. Default is `0.3`.

    R_thick : float, optional
        The scale radius of the thin disk in kpc. Default is `2.0`.

    Z_thick : float, optional
        The scale height of the thin disk in kpc. Default is `0.9`.

    f_thick : float, optional
        The fractional weight applied to the thick disk number density
        relative to the thin disk.
        Default is `0.04`.

    Rs_halo : float, optional
        The smoothing radius in kpc used to avoid singularities
        around the galactic center. Default is `1.0`.

    q_halo_ctr : float, optional
        The nominal oblateness of the halo at Galactic center.
        Default is `0.2`.

    q_halo_inf : float, optional
        The nominal oblateness of the halo infinitely far away.
        Default is `0.8`.

    r_q_halo : float, optional
        The scale radius over which the oblateness changes in kpc.
        Default is `6.`.

    eta_halo : float, optional
        The (negative) power law index describing the halo number density.
        Default is `4.2`.

    f_halo : float, optional
        The fractional weight applied to the halo number density.
        Default is `0.005`.

    feh_thin : float, optional
        The mean metallicity of the thin disk. Default is `-0.2`.

    feh_thin_sigma : float, optional
        The standard deviation in the metallicity of the thin disk.
        Default is `0.3`.

    feh_thick : float, optional
        The mean metallicity of the thick disk. Default is `-0.7`.

    feh_thick_sigma : float, optional
        The standard deviation in the metallicity of the thick disk.
        Default is `0.4`.

    feh_halo : float, optional
        The mean metallicity of the halo. Default is `-1.6`.

    feh_halo_sigma : float, optional
        The standard deviation in the metallicity of the halo.
        Default is `0.5`.

    max_age : float, optional
        The maximum allowed mean age (in Gyr). Default is `13.8`.

    min_age : float, optional
        The minimum allowed mean age (in Gyr). Default is `0.`.

    feh_age_ctr : float, optional
        The mean metallicity where the mean age is halfway between
        `max_age` and `min_age`. Default is `-0.5`.

    feh_age_scale : float, optional
        The exponential scale-length at which the mean age approaches
        `max_age` or `min_age` as it moves to lower or higher mean metallicity,
        respectively. Default is `0.5`.

    nsigma_from_max_age : float, optional
        The number of sigma away the mean age should be from `max_age`
        (i.e. the mean age is `nsigma_from_max_age`-sigma lower
        than `max_age`). Default is `2.`.

    max_sigma : float, optional
        The maximum allowed sigma (in Gyr). Default is `4.`.

    min_sigma : float, optional
        The minimum allowed sigma (in Gyr). Default is `1.`.

    return_components : bool, optional
        Whether to also return the separate components that make up
        the prior. Default is `False`.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(prior).

    components : dict, optional
        The individual components of `lnprior`.

    """

    # Compute volume factor.
    vol_factor = 2. * np.log(dists + 1e-300)  # dV = r**2 factor

    # Convert from observer-based coordinates to galactocentric cylindrical
    # coordinates.
    ell, b = np.full_like(dists, coord[0]), np.full_like(dists, coord[1])
    coords = SkyCoord(l=ell * units.deg, b=b * units.deg,
                      distance=dists * units.kpc,
                      frame='galactic')
    coords_cyl = coords.galactocentric.cartesian.represent_as(CylRep)
    R, Z = coords_cyl.rho.value, coords_cyl.z.value  # radius and height

    # Get thin disk component.
    logp_thin = logn_disk(R, Z, R_solar=R_solar, Z_solar=Z_solar,
                          R_scale=R_thin, Z_scale=Z_thin)
    logp_thin += vol_factor

    # Get thick disk component.
    logp_thick = logn_disk(R, Z, R_solar=R_solar, Z_solar=Z_solar,
                           R_scale=R_thick, Z_scale=Z_thick)
    logp_thick += vol_factor + np.log(f_thick)

    # Get halo component.
    logp_halo = logn_halo(R, Z, R_solar=R_solar, Z_solar=Z_solar,
                          R_smooth=Rs_halo, eta=eta_halo,
                          q_ctr=q_halo_ctr, q_inf=q_halo_inf, r_q=r_q_halo)
    logp_halo += vol_factor + np.log(f_halo)

    # Compute log-probability.
    lnprior = logsumexp([logp_thin, logp_thick, logp_halo], axis=0)

    # Collect components.
    components = {}
    components['number_density'] = [logp_thin, logp_thick, logp_halo]

    # Apply more sophisticated priors.
    if labels is not None:

        # Compute component membership probabilities.
        lnprior_thin = logp_thin - lnprior
        lnprior_thick = logp_thick - lnprior
        lnprior_halo = logp_halo - lnprior

        # Apply the galactic metallicity prior.
        try:
            # Grab metallicities.
            feh = labels['feh']

            # Compute "thin disk" metallicity prior.
            feh_lnp_thin = logp_feh(feh, feh_mean=feh_thin,
                                    feh_sigma=feh_thin_sigma)
            feh_lnp_thin += lnprior_thin

            # Compute "thick disk" metallicity prior.
            feh_lnp_thick = logp_feh(feh, feh_mean=feh_thick,
                                     feh_sigma=feh_thick_sigma)
            feh_lnp_thick += lnprior_thick

            # Compute halo metallicity prior.
            feh_lnp_halo = logp_feh(feh, feh_mean=feh_halo,
                                    feh_sigma=feh_halo_sigma)
            feh_lnp_halo += lnprior_halo

            # Compute total metallicity prior.
            feh_lnp = logsumexp([feh_lnp_thin, feh_lnp_thick, feh_lnp_halo],
                                axis=0)

            # Add to computed log-prior components.
            lnprior += feh_lnp
            components['feh'] = [feh_lnp_thin, feh_lnp_thick, feh_lnp_halo]
        except:
            pass

        # Apply the galactic age prior.
        try:
            # Grab ages (in Gyr).
            age = 10**labels['loga'] / 1e9
            nsig = nsigma_from_max_age

            # Compute thin disk age prior.
            age_lnp_thin = logp_age_from_feh(age, feh_mean=feh_thin,
                                             max_age=max_age, min_age=min_age,
                                             feh_age_ctr=feh_age_ctr,
                                             feh_age_scale=feh_age_scale,
                                             nsigma_from_max_age=nsig,
                                             max_sigma=max_sigma,
                                             min_sigma=min_sigma)
            age_lnp_thin += lnprior_thin

            # Compute thick disk age prior.
            age_lnp_thick = logp_age_from_feh(age, feh_mean=feh_thick,
                                              max_age=max_age, min_age=min_age,
                                              feh_age_ctr=feh_age_ctr,
                                              feh_age_scale=feh_age_scale,
                                              nsigma_from_max_age=nsig,
                                              max_sigma=max_sigma,
                                              min_sigma=min_sigma)
            age_lnp_thick += lnprior_thick

            # Compute halo age prior.
            age_lnp_halo = logp_age_from_feh(age, feh_mean=feh_halo,
                                             max_age=max_age, min_age=min_age,
                                             feh_age_ctr=feh_age_ctr,
                                             feh_age_scale=feh_age_scale,
                                             nsigma_from_max_age=nsig,
                                             max_sigma=max_sigma,
                                             min_sigma=min_sigma)
            age_lnp_halo += lnprior_halo

            # Compute total age prior.
            age_lnp = logsumexp([age_lnp_thin, age_lnp_thick, age_lnp_halo],
                                axis=0)

            # Add to computed log-prior components.
            lnprior += age_lnp
            components['age'] = [age_lnp_thin, age_lnp_thick, age_lnp_halo]
        except:
            pass

    if not return_components:
        return lnprior
    else:
        return lnprior, components


def dust_lnprior(dists, coord, avs, dustfile='bayestar2019_v1.h5',
                 offset=0., scale=1., smooth=1., scatter=0.2,
                 return_components=False):
    """
    Log-prior for a 3-D galactic dust model.

    Parameters
    ----------
    dists : `~numpy.ndarray` of shape `(N,)`
        Distance from the observer in kpc.

    coord : 2-tuple
        The `(l, b)` galaxy coordinates of the object.

    avs : `~numpy.ndarray` of shape `(N,)`
        Reddenings associated with the corresponding distance draws.

    dustfile : str, optional
        The filepath pointing to where the 3-D dust map is stored. The default
        is `bayestar2019_v1.h5`, which can be downloaded from the
        GitHub repository and is based on the "Bayestar19" 3-D
        dust map from Green et al. (2019).

    offset : float, optional
        A systematic offset to be added to the corresponding A(V) predictions
        from the dust map. Default is `0.` (no offset). Applied *after*
        `scale` is applied.

    scale : float, optional
        A systematic scaling to be multiplied to the corresponding A(V)
        predictions from the dust map. Default is `1.` (same scaling).
        Applied *before* `offset` is applied.

    smooth : float, optional
        The factor used to inflate the derived uncertainties and smooth out
        the dust map. Applied *before* `scatter` is added.
        Default is `1.`.

    scatter : float, optional
        Additional scatter that is added in quadrature with the derived
        uncertainties (*after* `smooth` has been applied)
        from the 3-D dust map. Default is `0.2`.

    return_components : bool, optional
        Whether to also return the components of the reddening LOS fit.
        Default is `False`.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(prior).

    av_mean : `~numpy.ndarray` of shape (N), optional
        The corresponding mean reddening of the prior.

    av_err : `~numpy.ndarray` of shape (N), optional
        The corresponding reddening error of the prior.

    """

    global bayestar
    try:
        # Get (mean, standard deviation) from `bayestar` object along LOS.
        av_dist, av_mean, av_err = bayestar.query(coord)
    except:
        # Load dustmap.
        bayestar = Bayestar(dustfile=dustfile)
        # Get (mean, standard deviation) from `bayestar` object along LOS.
        av_dist, av_mean, av_err = bayestar.query(coord)

    # Check that we have coverage.
    if np.all(np.isfinite(av_mean) & np.isfinite(av_err)):
        # Interpolate at corresponding distances.
        av_mean = scale * np.interp(dists, av_dist, av_mean) + offset
        av_err = smooth * scale * np.interp(dists, av_dist, av_err)
        av_err = np.sqrt(av_err**2 + scatter**2)

        # Evaluate Gaussian prior.
        chi2 = (avs - av_mean)**2 / av_err**2  # chi2
        lnorm = np.log(2. * np.pi * av_err**2)  # normalization
        lnprior = -0.5 * (chi2 + lnorm)
    else:
        # If we have no coverage, assume a uniform prior everywhere.
        lnprior = np.zeros_like(avs)

    if not return_components:
        return lnprior
    else:
        return lnprior, (av_mean, av_err)


def bin_pdfs_distred(data, cdf=False, ebv=False, dist_type='distance_modulus',
                     lndistprior=None, coord=None,
                     avlim=(0., 6.), rvlim=(1., 8.),
                     parallaxes=None, parallax_errors=None, Nr=100,
                     bins=(750, 300), span=None, smooth=0.01, rstate=None,
                     verbose=False):
    """
    Generate binned versions of the 2-D posteriors for the distance and
    reddening.

    Parameters
    ----------
    data : 3-tuple or 4-tuple containing `~numpy.ndarray`s of shape `(Nsamps)`
        The data that will be plotted. Either a collection of
        `(dists, reds, dreds)` that were saved, or a collection of
        `(scales, avs, rvs, covs_sar)` that will be used to regenerate
        `(dists, reds)` in conjunction with any applied distance
        and/or parallax priors.

    cdf : bool, optional
        Whether to compute the CDF along the reddening axis instead of the
        PDF. Useful when evaluating the MAP LOS fit. Default is `False`.

    ebv : bool, optional
        If provided, will convert from Av to E(B-V) when plotting using
        the provided Rv values. Default is `False`.

    dist_type : str, optional
        The distance format to be plotted. Options include `'parallax'`,
        `'scale'`, `'distance'`, and `'distance_modulus'`.
        Default is `'distance_modulus`.

    lndistprior : func, optional
        The log-distsance prior function used. If not provided, the galactic
        model from Green et al. (2014) will be assumed.

    coord : 2-tuple, optional
        The galactic `(l, b)` coordinates for the object, which is passed to
        `lndistprior` when re-generating the fits.

    avlim : 2-tuple, optional
        The Av limits used to truncate results. Default is `(0., 6.)`.

    rvlim : 2-tuple, optional
        The Rv limits used to truncate results. Default is `(1., 8.)`.

    parallaxes : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The parallax estimates for the sources.

    parallax_errors : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The parallax errors for the sources.

    Nr : int, optional
        The number of Monte Carlo realizations used when sampling using the
        provided parallax prior. Default is `100`.

    bins : int or list of ints with length `(ndim,)`, optional
        The number of bins to be used in each dimension. Default is `300`.

    span : iterable with shape `(ndim, 2)`, optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds. If not provided, the x-axis will use the
        provided Av bounds while the y-axis will span `(4., 19.)` in
        distance modulus (both appropriately transformed).

    smooth : float or list of floats with shape `(ndim,)`, optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 2-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.01` (1% smoothing).

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    verbose : bool, optional
        Whether to print progress to `~sys.stderr`. Default is `False`.

    Returns
    -------
    binned_vals : `~numpy.ndarray` of shape `(Nobj, Nxbin, Nybin)`
        Binned versions of the PDFs or CDFs.

    xedges : `~numpy.ndarray` of shape `(Nxbin+1,)`
        The edges defining the bins in distance.

    yedges : `~numpy.ndarray` of shape `(Nybin+1,)`
        The edges defining the bins in reddening.

    """

    # Initialize values.
    nobjs, nsamps = data[0].shape
    if rstate is None:
        try:
            # Attempt to use intel-specific version.
            rstate = np.random_intel
        except:
            # Fall back to default if not present.
            rstate = np.random
    if lndistprior is None:
        lndistprior = gal_lnprior
    if parallaxes is None:
        parallaxes = np.full(nobjs, np.nan)
    if parallax_errors is None:
        parallax_errors = np.full(nobjs, np.nan)

    # Set up bins.
    if dist_type not in ['parallax', 'scale', 'distance', 'distance_modulus']:
        raise ValueError("The provided `dist_type` is not valid.")
    if span is None:
        avlims = avlim
        dlims = 10**(np.array([4., 19.]) / 5. - 2.)
    else:
        avlims, dlims = span
    try:
        xbin, ybin = bins
    except:
        xbin = ybin = bins
    if ebv:
        ylims = avlims  # default Rv goes from [1., 8.] -> min(Rv) = 1.
    else:
        ylims = avlims
    if dist_type == 'scale':
        xlims = (1. / dlims[::-1])**2
    elif dist_type == 'parallax':
        xlims = 1. / dlims[::-1]
    elif dist_type == 'distance':
        xlims = dlims
    elif dist_type == 'distance_modulus':
        xlims = 5. * np.log10(dlims) + 10.
    xbins = np.linspace(xlims[0], xlims[1], xbin + 1)
    ybins = np.linspace(ylims[0], ylims[1], ybin + 1)
    dx, dy = xbins[1] - xbins[0], ybins[1] - ybins[0]
    xspan, yspan = xlims[1] - xlims[0], ylims[1] - ylims[0]

    # Set smoothing.
    try:
        if smooth[0] < 1:
            xsmooth = smooth[0] * xspan
        else:
            xsmooth = smooth[0] * dx
        if smooth[1] < 1:
            ysmooth = smooth[1] * yspan
        else:
            ysmooth = smooth[1] * dy
    except:
        if smooth < 1:
            xsmooth, ysmooth = smooth * xspan, smooth * yspan
        else:
            xsmooth, ysmooth = smooth * dx, smooth * dy

    # Compute binned PDFs.
    binned_vals = np.zeros((nobjs, xbin, ybin), dtype='float32')
    try:
        # Grab (distance, reddening (Av), differential reddening (Rv)) samples.
        ddraws, adraws, rdraws = copy.deepcopy(data)
        pdraws = 1. / ddraws
        sdraws = pdraws**2
        dmdraws = 5. * np.log10(ddraws) + 10.

        # Grab relevant draws.
        ydraws = adraws
        if ebv:
            ydraws /= rdraws
        if dist_type == 'scale':
            xdraws = sdraws
        elif dist_type == 'parallax':
            xdraws = pdraws
        elif dist_type == 'distance':
            xdraws = ddraws
        elif dist_type == 'distance_modulus':
            xdraws = dmdraws

        # Bin draws.
        for i, (xs, ys) in enumerate(zip(xdraws, ydraws)):
            # Print progress.
            if verbose:
                sys.stderr.write('\rBinning object {0}/{1}'
                                 .format(i + 1, nobjs))
            H, xedges, yedges = np.histogram2d(xs, ys, bins=(xbins, ybins))
            binned_vals[i] = H / nsamps
    except:
        # Regenerate distance and reddening samples from inputs.
        scales, avs, rvs, covs_sar = copy.deepcopy(data)

        if lndistprior is None and coord is None:
            raise ValueError("`coord` must be passed if the default distance "
                             "prior was used.")

        # Generate parallax and Av realizations.
        for i, stuff in enumerate(zip(scales, avs, rvs, covs_sar,
                                      parallaxes, parallax_errors, coord)):
            (scales_obj, avs_obj, rvs_obj, covs_sar_obj,
             parallax, parallax_err, crd) = stuff

            # Print progress.
            if verbose:
                sys.stderr.write('\rBinning object {0}/{1}'
                                 .format(i + 1, nobjs))

            # Draw random samples.
            sdraws, adraws, rdraws = draw_sar(scales_obj, avs_obj, rvs_obj,
                                              covs_sar_obj, ndraws=Nr,
                                              avlim=avlim, rvlim=rvlim,
                                              rstate=rstate)
            pdraws = np.sqrt(sdraws)
            ddraws = 1. / pdraws
            dmdraws = 5. * np.log10(ddraws) + 10.

            # Re-apply distance and parallax priors to realizations.
            lnp_draws = lndistprior(ddraws, crd)
            if parallax is not None and parallax_err is not None:
                lnp_draws += parallax_lnprior(pdraws, parallax, parallax_err)
            lnp = logsumexp(lnp_draws, axis=1)
            weights = np.exp(lnp_draws - lnp[:, None])
            weights /= weights.sum(axis=1)[:, None]
            weights = weights.flatten()

            # Grab draws.
            ydraws = adraws.flatten()
            if ebv:
                ydraws /= rdraws.flatten()
            if dist_type == 'scale':
                xdraws = sdraws.flatten()
            elif dist_type == 'parallax':
                xdraws = pdraws.flatten()
            elif dist_type == 'distance':
                xdraws = ddraws.flatten()
            elif dist_type == 'distance_modulus':
                xdraws = dmdraws.flatten()

            # Generate 2-D histogram.
            H, xedges, yedges = np.histogram2d(xdraws, ydraws,
                                               bins=(xbins, ybins),
                                               weights=weights)
            binned_vals[i] = H / nsamps

    # Apply smoothing.
    for i, (H, parallax, parallax_err) in enumerate(zip(binned_vals,
                                                        parallaxes,
                                                        parallax_errors)):
        # Establish minimum smoothing in distance.
        p1sig = np.array([parallax + parallax_err,
                          max(parallax - parallax_err, 1e-10)])
        if dist_type == 'scale':
            x_min_smooth = abs(np.diff(p1sig**2)) / 2.
        elif dist_type == 'parallax':
            x_min_smooth = abs(np.diff(p1sig)) / 2.
        elif dist_type == 'distance':
            x_min_smooth = abs(np.diff(1. / p1sig)) / 2.
        elif dist_type == 'distance_modulus':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore bad values
                x_min_smooth = abs(np.diff(5. * np.log10(1. / p1sig))) / 2.
        if np.isfinite(x_min_smooth):
            xsmooth_t = min(x_min_smooth, xsmooth)
        else:
            xsmooth_t = xsmooth
        try:
            xsmooth_t = xsmooth_t[0]  # catch possible list
        except:
            pass
        # Smooth 2-D PDF.
        binned_vals[i] = norm_kde(H, (xsmooth_t / dx, ysmooth / dy))

    # Compute CDFs.
    if cdf:
        for i, H in enumerate(binned_vals):
            binned_vals[i] = H.cumsum(axis=0)

    return binned_vals, xedges, yedges
