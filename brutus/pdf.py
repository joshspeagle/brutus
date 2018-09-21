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
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter as norm_kde
import copy

from .utils import draw_sav

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["imf_lnprior", "ps1_MrLF_lnprior", "parallax_lnprior",
           "logn_disk", "logn_halo", "logp_feh_disk", "logp_feh_halo",
           "gal_lnprior", "bin_pdfs_distred"]


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
        grid_Mr, grid_lnp = np.loadtxt(path+'/PSMrLF_lnprior.dat').T
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


def logp_feh_disk(feh, Z, feh_mean=-0.89, feh_sigma=0.2,
                  exp_amp=0.55, exp_scale=0.5):
    """
    Log-prior for the metallicity in the disk component of the galaxy.

    Parameters
    ----------
    feh : `~numpy.ndarray` of shape (N)
        The metallicities of the corresponding models whose `Z` has been
        provided.

    Z : `~numpy.ndarray` of shape (N)
        The height above the galactic midplane.

    feh_mean : float, optional
        The mean metallicity of the disk in the midplane. Default is `-0.89`.

    feh_sigma : float, optional
        The standard deviation in the metallicity of the disk.
        Default is `0.2`.

    exp_amp : float, optional
        The amplitude of the exponential dependence of the metallicity prior
        on `Z`. Default is `0.55`.

    exp_scale : float, optional
        The scale height for the exponential dependence of the metallicity
        prior on `Z` (in units of kpc). Default is `0.5`.

    Returns
    -------
    logn : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(number density).

    """

    # Compute mean metallicity.
    feh_mean_pred = feh_mean + exp_amp * np.exp(-np.abs(Z) / exp_scale)

    # Compute log-probability.
    chi2 = (feh_mean_pred - feh)**2 / feh_sigma**2  # chi2
    lnorm = np.log(2. * np.pi * feh_sigma**2)  # normalization
    lnprior = -0.5 * (chi2 + lnorm)

    return lnprior


def logp_feh_halo(feh, feh_mean=-1.46, feh_sigma=0.3):
    """
    Log-prior for the metallicity in the halo component of the galaxy.

    Parameters
    ----------
    feh : `~numpy.ndarray` of shape (N)
        The metallicities of the corresponding models whose `Z` has been
        provided.

    feh_mean : float, optional
        The mean metallicity of the halo. Default is `-1.46`.

    feh_sigma : float, optional
        The standard deviation in the metallicity of the halo.
        Default is `0.3`.

    Returns
    -------
    logn : `~numpy.ndarray` of shape (N)
        The corresponding normalized ln(number density).

    """

    # Compute log-probability.
    chi2 = (feh_mean - feh)**2 / feh_sigma**2  # chi2
    lnorm = np.log(2. * np.pi * feh_sigma**2)  # normalization
    lnprior = -0.5 * (chi2 + lnorm)

    return lnprior


def gal_lnprior(dists, coord, labels=None, R_solar=8., Z_solar=0.025,
                R_thin=2.15, Z_thin=0.245, R_thick=3.26, Z_thick=0.745,
                f_thick=0.13, Rb_halo=27.8, Rs_halo=0.5, q_halo=0.7,
                eta_halo_inner=2.62, eta_halo_outer=3.80, f_halo=0.003,
                feh_thin=-0.89, feh_thick=-0.75, feh_disk_sigma=0.2,
                feh_disk_exp_amp=0.55, feh_disk_exp_scale=0.5, feh_f_thin=0.63,
                feh_halo=-1.46, feh_halo_sigma=0.3,
                return_components=False):
    """
    Log-prior for a galactic model containing a thin disk, thick disk, and
    halo. The default behavior imposes a prior based on the total
    number density from all three components. If the metallicity is
    provided, then an associated galactic metallicity model is also imposed.
    Parameters taken from Green et al. (2014, 2015, 2018).

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

    feh_thin : float, optional
        The mean metallicity of the thin disk in the midplane.
        Default is `-0.89`.

    feh_thick : float, optional
        The mean metallicity of the thick disk in the midplane.
        Default is `-0.75`.

    feh_disk_sigma : float, optional
        The standard deviation in the metallicity of the disk.
        Default is `0.2`.

    feh_disk_exp_amp : float, optional
        The amplitude of the exponential dependence of the metallicity prior
        on `Z` in the disk. Default is `0.55`.

    feh_disk_exp_scale : float, optional
        The scale height for the exponential dependence of the metallicity
        prior on `Z` in the disk (in units of kpc). Default is `0.5`.

    feh_halo : float, optional
        The mean metallicity of the halo. Default is `-1.46`.

    feh_halo_sigma : float, optional
        The standard deviation in the metallicity of the halo.
        Default is `0.3`.

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

    # Collect components.
    components = [logp_thin, logp_thick, logp_halo]

    # Apply the galactic metallicity prior.
    if labels is not None:
        try:
            # Grab metallicities.
            feh = labels['feh']

            # Compute membership probabilities.
            logp_disk = logsumexp([logp_thin, logp_thick], axis=0)
            lnprior_disk = logp_disk - lnprior
            lnprior_halo = logp_halo - lnprior

            # Compute think disk metallicity prior.
            feh_lnp_thin = logp_feh_disk(feh, Z, feh_mean=feh_thin,
                                         feh_sigma=feh_disk_sigma,
                                         exp_amp=feh_disk_exp_amp,
                                         exp_scale=feh_disk_exp_scale)
            feh_lnp_thin += np.log(feh_f_thin) + lnprior_disk

            # Compute thick disk metallicity prior.
            feh_lnp_thick = logp_feh_disk(feh, Z, feh_mean=feh_thick,
                                          feh_sigma=feh_disk_sigma,
                                          exp_amp=feh_disk_exp_amp,
                                          exp_scale=feh_disk_exp_scale)
            feh_lnp_thick += np.log(1. - feh_f_thin) + lnprior_disk

            # Compute halo metallicity prior.
            feh_lnp_halo = logp_feh_halo(feh, feh_mean=feh_halo,
                                         feh_sigma=feh_halo_sigma)
            feh_lnp_halo += lnprior_halo

            # Compute total metallicity prior.
            feh_lnp = logsumexp([feh_lnp_thin, feh_lnp_thick, feh_lnp_halo],
                                axis=0)

            # Add to computed log-prior components.
            lnprior += feh_lnp
            components += [feh_lnp_thin, feh_lnp_thick, feh_lnp_halo]
        except:
            raise ValueError("No `'feh'` provided -- cannot compute "
                             "galactic metallicity prior.")

    if not return_components:
        return lnprior
    else:
        return lnprior, components


def bin_pdfs_distred(data, cdf=False, Rv=None, dist_type='distance_modulus',
                     lndistprior=None, coords=None, avlim=(0., 6.),
                     parallaxes=None, parallax_errors=None, Nr=100,
                     bins=(750, 300), span=None, smooth=0.01, rstate=None,
                     verbose=False):
    """
    Generate binned versions of the 2-D posteriors for the distance and
    reddening.

    Parameters
    ----------
    data : 2-tuple or 3-tuple containing `~numpy.ndarray`s of shape `(Nsamps)`
        The data that will be plotted. Either a collection of `(dists, reds)`
        that were saved, or a collection of `(scales, avs, covs_sa)` that
        will be used to regenerate `(dists, reds)` in conjunction with
        any applied ditsance and/or parallax priors.

    cdf : bool, optional
        Whether to compute the CDF along the reddening axis instead of the
        PDF. Useful when evaluating the MAP LOS fit. Default is `False`.

    Rv : float, optional
        If provided, will convert from Av to E(B-V). Default is `None`.

    dist_type : str, optional
        The distance format to be plotted. Options include `'parallax'`,
        `'scale'`, `'distance'`, and `'distance_modulus'`.
        Default is `'distance_modulus`.

    lndistprior : func, optional
        The log-distsance prior function used. If not provided, the galactic
        model from Green et al. (2014) will be assumed.

    coord : iterable of 2-tuples with len `Nobj`, optional
        The galactic `(l, b)` coordinates for the object, which is passed to
        `lndistprior`.

    avlim : 2-tuple, optional
        The Av limits used to truncate results. Default is `(0., 6.)`.

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
    if Rv is not None:
        ylims = np.array(avlim) / Rv
    else:
        ylims = avlim
    if dist_type == 'scale':
        xlims = (1. / dlims[::-1])**2
    elif dist_type == 'parallax':
        xlims = 1. / dlims[::-1]
    elif dist_type == 'distance':
        xlims = dlims
    elif dist_type == 'distance_modulus':
        xlims = 5. * np.log10(dlims) + 10.
    xbins = np.linspace(xlims[0], xlims[1], xbin+1)
    ybins = np.linspace(ylims[0], ylims[1], ybin+1)
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
        # Grab distance and reddening samples.
        ddraws, adraws = copy.deepcopy(data)
        pdraws = 1. / ddraws
        sdraws = pdraws**2
        dmdraws = 5. * np.log10(ddraws) + 10.

        # Grab relevant draws.
        ydraws = adraws
        if Rv is not None:
            ydraws /= Rv
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
                sys.stderr.write('\rBinning object {0}/{1}'.format(i+1, nobjs))
            H, xedges, yedges = np.histogram2d(xs, ys, bins=(xbins, ybins))
            binned_vals[i] = H / nsamps
    except:
        # Regenerate distance and reddening samples from inputs.
        scales, avs, covs_sa = copy.deepcopy(data)

        if lndistprior is None and coord is None:
            raise ValueError("`coord` must be passed if the default distance "
                             "prior was used.")

        # Generate parallax and Av realizations.
        for i, stuff in enumerate(zip(scales, avs, covs_sa,
                                      parallaxes, parallax_errors,
                                      coords)):
            (scales_obj, avs_obj, covs_sa_obj,
             parallax, parallax_err, coord) = stuff

            # Print progress.
            if verbose:
                sys.stderr.write('\rBinning object {0}/{1}'.format(i+1, nobjs))

            # Draw random samples.
            sdraws, adraws = draw_sav(scales_obj, avs_obj, covs_sa_smooth_obj,
                                      ndraws=Nr, avlim=avlim, rstate=rstate)
            pdraws = np.sqrt(sdraws)
            ddraws = 1. / pdraws
            dmdraws = 5. * np.log10(ddraws) + 10.

            # Re-apply distance and parallax priors to realizations.
            lnp_draws = lndistprior(ddraws, coord)
            if parallax is not None and parallax_err is not None:
                lnp_draws += parallax_lnprior(pdraws, parallax, parallax_err)
            lnp = logsumexp(lnp_draws, axis=1)
            weights = np.exp(lnp_draws - lnp[:, None])
            weights /= weights.sum(axis=1)[:, None]
            weights = weights.flatten()

            # Grab draws.
            ydraws = adraws.flatten()
            if Rv is not None:
                ydraws /= Rv
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
                warnings.simplefilter("ignore")
                x_min_smooth = abs(np.diff(5. * np.log10(1. / p1sig))) / 2.
        if np.isfinite(x_min_smooth):
            xsmooth_t = min(x_min_smooth, xsmooth)
        else:
            xsmooth_t = xsmooth
        # Smooth 2-D PDF.
        binned_vals[i] = norm_kde(H, (xsmooth_t / dx, ysmooth / dy))

    # Compute CDFs.
    if cdf:
        for i, H in enumerate(binned_vals):
            binned_vals[i] = H.cumsum(axis=0)

    return binned_vals, xedges, yedges
