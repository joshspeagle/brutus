#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF functions.

"""

from __future__ import (print_function, division)
import six
from six.moves import range

import sys
import os
import warnings
import math
import numpy as np
import warnings
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates import CylindricalRepresentation as CylRep

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["imf_lnprior", "parallax_lnprior",
           "logn_disk", "logn_halo", "gal_lnprior"]


def imf_lnprior(mgrid):
    """
    Apply Kroupa IMF prior over the provided initial mass grid.

    Parameters
    ----------
    mgrid : `~numpy.ndarray` of shape (Ngrid)
        Grid of initial mass the Kroupa IMF will be evaluated over.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (Ngrid)
        The corresponding unnormalized ln(prior).

    """

    # Initialize log-prior.
    lnprior = np.zeros_like(mgrid)

    # Low mass.
    low_mass = mgrid <= 0.08
    lnprior[low_mass] = -0.3 * np.log(mgrid[low_mass])

    # Intermediate mass.
    mid_mass = (mgrid <= 0.5) & (mgrid > 0.08)
    lnprior[mid_mass] = -1.3 * np.log(mgrid[mid_mass]) + np.log(0.08)

    # High mass.
    high_mass = mgrid > 0.5
    lnprior[high_mass] = -2.3 * np.log(mgrid[high_mass]) + np.log(0.5 * 0.08)

    return lnprior


def parallax_lnprior(parallaxes, p_meas, p_err):
    """
    Apply parallax prior using a measured parallax.

    Parameters
    ----------
    parallaxes : `~numpy.ndarray` of shape (N)
        Parallaxes.

    p_meas : float, optional
        Measured parallax. Default is `0.`.

    p_std : float, optional
        Measured parallax error. Default is `1e10`.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (Ngrid)
        The corresponding ln(prior).

    """

    if np.isfinite(p_meas) and np.isfinite(p_err):
        # Compute log-prior.
        chi2 = (parallaxes - p_meas)**2 / p_err**2  # chi2
        lnorm = np.log(2. * np.pi * p_err**2)  # normalization
        lnprior = -0.5 * (chi2 + lnorm)
    else:
        # If no measurement, assume a uniform prior everywhere.
        lnprior = np.zeros_like(parallaxes) - np.log(len(parallaxes))

    return lnprior


def logn_disk(R, Z, R_solar=8., Z_solar=0.025, R_scale=2.15, Z_scale=0.245):
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
        Default is `8.`.

    Z_solar : float, optional
        The solar height above the galactic midplane in kpc.
        Default is `0.025`.

    R_scale : float, optional
        The scale radius of the disk in kpc. Default is `2.15`.

    Z_scale : float, optional
        The scale height of the disk in kpc. Default is `0.245`.

    Returns
    -------
    logn : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(number density).

    """

    rterm = (R - R_solar) / R_scale  # radius term
    zterm = (np.abs(Z) - np.abs(Z_solar)) / Z_scale  # height term

    return -(rterm + zterm)


def logn_halo(R, Z, R_solar=8., R_break=27.8, R_smooth=0.5, q_h=0.7,
              eta_inner=2.62, eta_outer=3.80):
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
        Default is `8.`.

    R_break : float, optional
        The radius in kpc at which the density switches from the inner
        component to the outer component. Default is `27.8`.

    R_smooth : float, optional
        The smoothing radius in kpc used to avoid singularities
        around the galactic center. Default is `0.5`.

    q_h : float, optional
        The oblateness of the halo measured from `[0., 1.]`.
        Default is `0.7`.

    eta_inner : float, optional
        The (negative) power law index describing the number density
        in the inner portion of the halo (i.e. `R <= R_break`).
        Default is `2.62`.

    eta_outer : float, optional
        The (negative) power law index describing the number density in
        the outer portion of the halo (i.e. `R > R_break`).
        Default is `3.80`.

    Returns
    -------
    logn : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(number density).

    """

    # Initialize value.
    logn = np.zeros_like(R)

    # Compute effective radius.
    Reff = np.sqrt(R**2 + (Z / q_h)**2 + R_smooth**2)

    # Compute inner component.
    inner = Reff <= R_break
    logn[inner] = -eta_inner * np.log(Reff[inner] / R_solar)

    # Compute outer component
    offset = (eta_outer - eta_inner) * np.log(R_break / R_solar)
    logn[~inner] = -eta_outer * np.log(Reff[~inner] / R_solar) + offset

    return logn


def gal_lnprior(dists, coord, R_solar=8., Z_solar=0.025,
                R_thin=2.15, Z_thin=0.245, R_thick=3.26, Z_thick=0.745,
                f_thick=0.13, Rb_halo=27.8, Rs_halo=0.5, q_halo=0.7,
                eta_halo_inner=2.62, eta_halo_outer=3.80, f_halo=0.003,
                return_components=False):
    """
    Log-prior in distance for a galactic model containing a thin disk,
    thick disk, and halo. Parameters taken from Green et al. (2014).

    Parameters
    ----------
    dists : `~numpy.ndarray` of shape (N)
        Distance from the observer in kpc.

    coord : 2-tuple
        The `(l, b)` galaxy coordinates of the object.

    R_solar : float, optional
        The solar distance from the center of the galaxy in kpc.
        Default is `8.`.

    Z : `~numpy.ndarray` of shape (N)
        The height above the galactic midplane.

    R_solar : float, optional
        The solar distance from the center of the galaxy in kpc.
        Default is `8.`.

    Z_solar : float, optional
        The solar height above the galactic midplane in kpc.
        Default is `0.025`.

    R_thin : float, optional
        The scale radius of the thin disk in kpc. Default is `2.15`.

    Z_thin : float, optional
        The scale height of the thin disk in kpc. Default is `0.245`.

    R_thick : float, optional
        The scale radius of the thin disk in kpc. Default is `3.26`.

    Z_thick : float, optional
        The scale height of the thin disk in kpc. Default is `0.745`.

    f_thick : float, optional
        The fractional weight applied to the thick disk number density.
        Default is `0.13`.

    Rb_halo : float, optional
        The radius in kpc at which the density switches from the inner
        component to the outer component in the halo. Default is `27.8`.

    Rs_halo : float, optional
        The smoothing radius in kpc used to avoid singularities
        around the galactic center. Default is `0.5`.

    q_halo : float, optional
        The oblateness of the halo measured from `[0., 1.]`.
        Default is `0.7`.

    eta_halo_inner : float, optional
        The (negative) power law index describing the number density
        in the inner portion of the halo (i.e. `R <= R_break`).
        Default is `2.62`.

    eta_halo_outer : float, optional
        The (negative) power law index describing the number density in
        the outer portion of the halo (i.e. `R > R_break`).
        Default is `3.80`.

    f_halo : float, optional
        The fractional weight applied to the halo number density.
        Default is `0.003`.

    return_components : bool, optional
        Whether to also return the separate components. Default is `False`.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(prior).

    lnprior_thin : `~numpy.ndarray` of shape (N), optional
        The corresponding normalized ln(prior) from the thin disk.

    lnprior_thick : `~numpy.ndarray` of shape (N), optional
        The corresponding normalized ln(prior) from the thick disk.

    lnprior_halo : `~numpy.ndarray` of shape (N), optional
        The corresponding normalized ln(prior) from the halo.

    """

    # Compute volume factor.
    vol_factor = 2. * np.log(dists + 1e-300)  # dV = r**2 factor

    # Convert from observer-based coordinates to galactocentric cylindrical
    # coordinates.
    l, b = np.full_like(dists, coord[0]), np.full_like(dists, coord[1])
    coords = SkyCoord(l=l*units.deg, b=b*units.deg, distance=dists*units.kpc,
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
    logp_halo = logn_halo(R, Z, R_solar=R_solar, R_break=Rb_halo,
                          R_smooth=Rs_halo, q_h=q_halo,
                          eta_inner=eta_halo_inner, eta_outer=eta_halo_outer)
    logp_halo += vol_factor + np.log(f_halo)

    # Compute log-probability.
    lnprior = logsumexp([logp_thin, logp_thick, logp_halo], axis=0)
    lnorm = logsumexp(lnprior)
    lnprior -= lnorm

    if not return_components:
        return lnprior
    else:
        logp_thin -= lnorm
        logp_thick -= lnorm
        logp_halo -= lnorm
        return lnprior, logp_thin, logp_thick, logp_halo
