#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Star cluster fitting utilities.

"""

from __future__ import (print_function, division)

import warnings
import numpy as np
from scipy.stats import chi2 as chisquare

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["isochrone_loglike"]


def isochrone_loglike(theta, isochrone, phot, err, cluster_params='free',
                      offsets='fixed', corr_params='fixed',
                      mini_bound=0.08, eep_binary_max=480.,
                      smf_grid=None, eep_grid=None,
                      parallax=None, parallax_err=None,
                      cluster_prob=0.95, dim_prior=True, return_lnls=False):
    """
    Compute the log-likelihood for a given co-eval stellar population
    based on the given isochrone parameters `theta`. Assumes a uniform
    outlier model in distance and parallax.

    Parameters
    ----------
    theta : `~numpy.ndarray` of shape `(Nparams,)`
        A collection of parameters used to compute the cluster model.
        The full collection of parameters that can be modeled includes three
        sets of parameters. The first are the parameters related to
        the **physical cluster properties**.
        The second are the **"offsets"** in each band (i.e. what the
        data need to be multiplied by to agree with the models).
        The third are the parameters used to apply **empirical corrections**
        to the isochrone as a function of mass, metallicity, and EEP.
        Certain parameters can be held fixed in the likelihood by specifying
        their value via `cluster_params`, `offsets`, and/or `corr_params`.
        In those cases, `theta` essentially "skips" over fixed parameters
        based on their relative position.
        As an example, if `feh`, `loga`, and `dist` are specified in
        `cluster_params`, the first few elements of `theta`
        would be `(av, rv, fout)` rather than `(feh, loga, av)`.
        Note that if `parallax` and `parallax_err` are not provided,
        *either* the `dist` value in `cluster_params` must be fixed *or*
        at least one entry in `offsets` must be fixed.

    isochrone : `~seds.Isochrone` object
        The `~seds.Isochrone` object from the `~seds` module used to generate
        the photometry and associated physical parameters.

    phot : `~numpy.ndarray` of shape `(Nobj, Nbands,)`
        Measured flux densities, in units of `10**(-0.4 * mag)`.

    err : `~numpy.ndarray` of shape `(Nobj, Nbands,)`
        Associated errors on the measurements in the same units.

    cluster_params : iterable of shape `(6,)` or `'free'`, optional
        The parameters of the cluster, which include (in order):
        metallicity (`feh`), log(age) (`loga`), reddening in A(V) (`av`),
        shape of the reddening curve in R(V) (`rv`), and the distance to the
        cluster in pc (`dist`). If `'free'` is passed, all values will be
        included when reading `theta`. If an iterable is passed, any
        non-`None` value will be considered fixed and not included
        when reading in parameters from `theta`. For example, fixing the
        log(age), distance, and R(V) would look like
        `[None, 9.5, None, 3.3, 880.]`. Default is `'free'`.

    offsets : iterable of shape `(Nfilt,)`, `'free'`, or `'fixed'`, optional
        Flux offsets that are *multiplied* to the observed fluxes
        to ensure better agreement with the models. If `'free'` is passed,
        all values will be included when reading `theta`. If `'fixed'` is
        passed, all values will be fixed to `1.0` and not included when reading
        `theta`. If an iterable is passed, any non-`None` value
        will be considered fixed and not included when reading in parameters
        from `theta`. An example of this might look like
        `[1.02, None, None, 0.98, 0.97]`. Default is `'fixed'`.

    corr_params : iterable of shape `(4,)` or `'fixed'`, optional
        Parameters used to apply empirical corrections to the isochrone
        as a function of mass, metallicity, and EEP. These include (in order):
        the slope of the change in log(Teff) as a function of
        mass (`dtdm`), the slope of the change in log(R) as a function of mass
        (`drdm`), the smoothing scale used to patch in corrections around
        the turnoff (`msto_smooth`), and the metallicity dependence
        (`feh_scale`). If `'fixed'` is passed, all values
        will be taken to be the defaults specified in `isochrone`
        and not included when reading `theta`. If an iterable is passed,
        any non-`None` value will be considered fixed and not included
        when reading in parameters from `theta`. For example, if we wanted
        to specify `msto_smooth` but leave the other parameters free, we would
        pass `[None, None, 30., None]`. Default is `'fixed'`.

    mini_bound : float, optional
        The lowest initial mass value down to which the isochrone is evaluated.
        Default is `0.08`.

    eep_binary_max : float, optional
        The maximum EEP where binaries are permitted. By default, binaries
        are disallowed once the primary begins its giant expansion
        after turning off the Main Sequence at `eep=480.`.

    smf_grid : `~numpy.ndarray` of shape `(Nsmf,)`, optional
        The grid of secondary mass fraction values over which to evaluate
        the isochrone. Default is the adaptively-spaced grid
        `[0., 0.2, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65,
          0.7, 0.75, 0.8, 0.85, 0.9, 1.0]`.

    eep_grid : `~numpy.ndarray` of shape `(Neep,)`, optional
        The grid of EEP values over which to evaluate
        the isochrone. If not provided, the default grid uses 2000
        equally-spaced points between `202.` and `808.`.

    parallax : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The measured parallaxes corresponding to each object.
        Note that if `parallax` and `parallax_err` are not provided,
        either the `dist` value in `cluster_params` must be fixed or
        at least one entry in `offsets` must be fixed.

    parallax_err : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The measurement errors corresponding to the values in `parallax`.
        Note that if `parallax` and `parallax_err` are not provided,
        either the `dist` value in `cluster_params` must be fixed or
        at least one entry in `offsets` must be fixed.

    cluster_prob : float or `~numpy.ndarray` of shape `(Nobj,)`, optional
        Cluster membership probabilities for each object. If not provided,
        defaults to `0.95`. These are multiplied by `1. - f_out`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood. Transforms the likelihood from a combination of a
        Normal distribution (inlier) with a uniform distribution (outlier)
        to a Chi-Square distribution (inlier) with `Ndof` degrees of freedom
        for each object, where `Ndof` is the total number of observed bands
        including the parallax, and a simple threshold (outlier).
        Default is `True`.

    return_lnls : bool, optional
        Whether to also return the `Nobj` log-likelihoods for each individual
        object. Default is `False`.

    Returns
    -------
    lnl : float
        The total log-likelihood.

    lnls : `~numpy.ndarray` of shape `(Nobj,)`, optional
        The log-likelihoods for each individual object. Returned if
        `return_lnls=True`.

    """

    # Initialize data.
    Nobjs, Nbands = phot.shape

    # Check isochrone object.
    if isochrone is None:
        raise ValueError("The `isochrone` object must be properly specified "
                         "or photometry cannot be properly generated!")
    iso = isochrone  # shorter alias

    # Check data.
    if phot is None:
        raise ValueError("The photometry must be provided to compute the "
                         "log-likelihood!")
    if err is None:
        raise ValueError("The errors on the photometry must be provided to "
                         "compute the log-likelihood!")
    phot_mask = np.isfinite(phot) & np.isfinite(err)  # band mask (1/0)
    phot_n = np.sum(phot_mask, axis=1)  # number of bands per object
    if np.any(np.all(~phot_mask, axis=1)):
        raise ValueError("At least one object has no valid data entries!")

    # Initialize secondary mass fraction (SMF) grid.
    if smf_grid is None:
        smf_grid = np.array([0., 0.2, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65,
                             0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    Nsmf = len(smf_grid)

    # Compute spacing on SMF grid.
    if len(smf_grid) > 1:
        grad_smf = np.gradient(smf_grid)
    else:
        grad_smf = np.array([1.])

    # Initialize EEP (becomes mass) grid.
    if eep_grid is None:
        eep_grid = np.linspace(202., 808., 2000)

    # Sanity check to make sure distance and offsets are not degenerate.
    if parallax is None and parallax_err is None:
        if offsets == 'free' and (cluster_params == 'free' or
                                  cluster_params[4] is None):
            raise ValueError("Without any measured parallaxes, there is a "
                             "degeneracy between the photometry offsets "
                             "and the distance. Please provide either a "
                             "distance value in `cluster_params` or at "
                             "least one offset in `offsets`.")

    # Sanity check to make sure either `feh_scale` or `drdm` and `dtdm` are
    # provided.
    if corr_params != 'fixed' and ((corr_params[0] is None or
                                    corr_params[1] is None) and
                                   corr_params[3] is None):
        raise ValueError("If `feh_scale` is not provided, then `dtdm` and "
                         "`drdm` must be fixed since the parameters are "
                         "perfectly degenerate.")

    # Parallax checks.
    if parallax is None and parallax_err is not None:
        raise ValueError("You forgot to provide the parallaxes to go along "
                         "with the errors!")
    if parallax is not None and parallax_err is None:
        raise ValueError("You forgot to provide the parallax errors to go "
                         "along with the parallaxes!")

    # Read in `theta`.
    counter = 0

    # Read in cluster parameters.
    if cluster_params == 'free':
        # If no constraints are provided, read out list of all parameters.
        feh, loga, av, rv, dist, fout = theta[:6]
        counter += 6
    else:
        # If constraints are specified, we read out parameters one at a time.
        p = np.zeros(6)
        for i, c in enumerate(cluster_params):
            if c is None:
                # If parameter is not specified, read out from `theta` and
                # increment the counter.
                p[i] = theta[counter]
                counter += 1
            else:
                # If parameter is specified, set value and do not increment
                # counter.
                p[i] = c
        feh, loga, av, rv, dist, fout = np.array(p)
    fout = max(min(1., fout), 0.)  # bound between [0., 1.]

    # Read in multiplicative offsets.
    if offsets == 'free':
        # If no constraints are provided, read out list of all offsets.
        Xb = theta[counter:counter + Nbands]
        counter += Nbands
    elif offsets == 'fixed':
        Xb = np.ones(Nbands)
        counter += Nbands
    else:
        # If constraints are specified, we read out parameters one at a time.
        Xb = np.zeros(Nbands)
        for i, z in enumerate(offsets):
            if z is None:
                # If offset is not specified, read out from `theta` and
                # increment the counter.
                Xb[i] = theta[counter]
                counter += 1
            else:
                # If offset is specified, set value and do not increment
                # counter.
                Xb[i] = z

    # Read in empirical correction parameters.
    if corr_params == 'fixed':
        corr_coef = None
        counter += 4
    else:
        # If constraints are specified, we read out parameters one at a time.
        p = np.zeros(4)
        for i, c in enumerate(corr_params):
            if c is None:
                # If parameter is not specified, read out from `theta` and
                # increment the counter.
                p[i] = theta[counter]
                counter += 1
            else:
                # If parameter is specified, set value and do not increment
                # counter.
                p[i] = c
        corr_coef = np.array(p)

    # Compute parallax contribution.
    chi2_p = np.zeros(Nobjs)
    lnorm_p = np.zeros(Nobjs)
    if parallax is not None and parallax_err is not None:
        parallax_mask = np.isfinite(parallax) & np.isfinite(parallax_err)
        chi2_p[parallax_mask] += ((parallax[parallax_mask] - 1e3 / dist)**2
                                  / parallax_err[parallax_mask]**2)
        lnorm_p[parallax_mask] += np.log(2. * np.pi *
                                         parallax_err[parallax_mask]**2)
        phot_n[parallax_mask] += 1

    # Compute outlier model.
    if dim_prior:
        # If using the chi-square distribution, cut on a "p-value" of 1e-5.
        outlier_chi2 = chisquare.ppf(1. - 1e-5, phot_n)  # chi2 threshold
        lnl_outlier = chisquare.logpdf(outlier_chi2, phot_n)  # chi2 log(prob)
    else:
        # If using a normal distribution, compute quasi-uniform outlier model.
        outlier_max = np.nanmax(phot + 3. * err, axis=0)
        outlier_min = np.nanmin(phot - 3. * err, axis=0)
        outlier_size = (6. * err) / (outlier_max - outlier_min)  # err / sides
        outlier_size[~phot_mask] = 1.  # remove nans
        outlier_vol = np.prod(outlier_size * phot_mask  # include sides w/ data
                              + 1. * ~phot_mask, axis=1)  # ignore w/o data
        if parallax is not None and parallax_err is not None:
            p_max = np.nanmax((parallax + 3. * parallax_err)[parallax_mask])
            p_min = np.nanmin((parallax - 3. * parallax_err)[parallax_mask])
            outlier_vol[parallax_mask] *= ((6. * parallax_err[parallax_mask]) /
                                           (p_max - p_min))
        lnl_outlier = np.log(1. / outlier_vol)  # log(uniform) PDF

    # Compute cluster membership probabilities (inlier vs outlier).
    ln_fin = np.log(cluster_prob * (1. - fout))
    ln_fout = np.log(1. - cluster_prob * (1. - fout))

    # Initialize log-likelihoods.
    lnls = np.full((Nsmf, Nobjs), -np.inf)

    # Initialize flag indicating whether models where binaries are disallowed
    # have already been computed. These are identical for any given SMF, and
    # so are ignored in subsequent computations.
    identical_models_computed = False

    # Loop over SMF grid.
    for i, smf in enumerate(smf_grid):

        # Generate isochrone.
        cmd_sed, params1, params2 = iso.get_seds(feh=feh, loga=loga,
                                                 av=av, rv=rv, eep=eep_grid,
                                                 smf=smf, dist=dist,
                                                 mini_bound=mini_bound,
                                                 eep_binary_max=eep_binary_max,
                                                 corr_params=corr_coef)

        # Get initial mass grid.
        cmd_mini = params1['mini']  # grid
        grad_mini = np.gradient(cmd_mini)  # spacing

        # Compute mask so we don't waste time evaluating non-existent
        # or repeated models.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            if identical_models_computed:
                # If the first set of models where binaries are disallowed
                # have been computed, mask out those models in addition to
                # non-existent values.
                cmd_mask = np.where(np.any(np.isfinite(cmd_sed), axis=1) &
                                    (grad_mini > 0.) &
                                    (eep_grid <= eep_binary_max))[0]
            else:
                # If this is the first time we're doing this, only mask out
                # non-existent values.
                cmd_mask = np.where(np.any(np.isfinite(cmd_sed), axis=1) &
                                    (grad_mini > 0.))[0]
                identical_models_computed = True

        # If at least one point on the isochrone exists, compute
        # the log-likelihood.
        Ncmd = len(cmd_mask)
        if Ncmd > 0:

            # Mask entries.
            cmd_sed = cmd_sed[cmd_mask]
            cmd_mini = cmd_mini[cmd_mask]
            grad_mini = grad_mini[cmd_mask]

            # Compute photometry contribution.
            phot_t, err_t = phot * Xb, err * Xb
            cmd_phot = 10**(-0.4 * cmd_sed)
            chi2_cmd = np.nansum((phot_t - cmd_phot[:, None])**2 / err_t**2,
                                 axis=-1)
            lnorm_cmd = np.nansum(np.log(2. * np.pi * err_t**2), axis=-1)

            # Compute log-likelihood.
            chi2 = chi2_cmd + chi2_p
            lnorm = lnorm_cmd + lnorm_p
            if dim_prior:
                # Compute logpdf of chi2 distribution.
                lnl_cmd = chisquare.logpdf(chi2, phot_n)
            else:
                # Compute logpdf of normal distribution.
                lnl_cmd = -0.5 * (chi2 + lnorm)
            lnl_cmd[~np.isfinite(lnl_cmd)] = -np.inf  # mask bad values

            # Compute mass prior.
            lnprior_mini = np.log(grad_mini)  # add contribution to integral

            # Compute SMF prior as a function of mass.
            lnprior_smf = np.log(grad_smf[i])

            # Integrate over mass.
            lnprior = lnprior_mini + lnprior_smf
            lnls[i] = logsumexp(lnl_cmd + lnprior[:, None], axis=0)

    # Integrate over SMF for each object.
    lnl = logsumexp(lnls, axis=0)

    # Apply outlier mixture model.
    lnl_mix = np.logaddexp(lnl + ln_fin,
                           lnl_outlier + ln_fout)

    # Compute total log-likelihood
    lnl_tot = np.sum(lnl_mix)

    if return_lnls:
        return lnl_tot, lnl_mix
    else:
        return lnl_tot
