#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collection of methods that generate the underlying grid of SED models. Code
also contributed by Ben Johnson and Phil Cargile.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import warnings
import numpy as np
import time
from copy import deepcopy
from itertools import product
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy import polyfit
from scipy.optimize import minimize

from .filters import FILTERS
from .utils import add_mag

# Rename parameters from what is in the MIST HDF5 file.
# This makes it easier to use parameter names as keyword arguments.
rename = {"mini": "initial_mass",  # input parameters
          "eep": "EEP",
          "feh": "initial_[Fe/H]",
          "afe": "initial_[a/Fe]",
          "mass": "star_mass",  # outputs
          "feh_surf": "[Fe/H]",
          "afe_surf": "[a/Fe]",
          "loga": "log_age",
          "logt": "log_Teff",
          "logg": "log_g",
          "logl": "log_L",
          "logr": "log_R"}

__all__ = ["MISTtracks", "SEDmaker", "FastNN", "FastNNPredictor",
           "Isochrone"]


class MISTtracks(object):
    """
    An object that linearly interpolates the MIST tracks in EEP, initial mass,
    and metallicity. Uses `~scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    mistfile : str, optional
        The name of the HDF5 file containing the MIST tracks. Default is
        `MIST_1.2_EEPtrk.h5` and is extracted from `data/DATAFILES/`.

    predictions : iterable of shape `(4)`, optional
        The names of the parameters to output at the request location in
        the `labels` parameter space. Default is
        `["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"]`.
        **Do not modify this unless you know what you're doing.**

    ageweight : bool, optional
        Whether to compute the associated d(age)/d(EEP) weights at each
        EEP grid point, which are needed when applying priors in age.
        Default is `True`.

    verbose : bool, optional
        Whether to output progress to `~sys.stderr`. Default is `True`.

    """

    def __init__(self, mistfile=None,
                 predictions=["loga", "logl", "logt", "logg",
                              "feh_surf", "afe_surf"],
                 ageweight=True, verbose=True):

        labels = ["mini", "eep", "feh", "afe"]

        # Initialize values.
        self.labels = list(np.array(labels))
        self.predictions = list(np.array(predictions))
        self.ndim, self.npred = len(self.labels), len(self.predictions)
        self.null = np.zeros(self.npred) + np.nan

        # Set label references.
        self.mini_idx = np.where(np.array(self.labels) == 'mini')[0][0]
        self.eep_idx = np.where(np.array(self.labels) == 'eep')[0][0]
        self.feh_idx = np.where(np.array(self.labels) == 'feh')[0][0]
        self.logt_idx = np.where(np.array(self.predictions) == 'logt')[0][0]
        self.logl_idx = np.where(np.array(self.predictions) == 'logl')[0][0]
        self.logg_idx = np.where(np.array(self.predictions) == 'logg')[0][0]

        # Import MIST grid.
        if mistfile is None:
            mistfile = 'data/DATAFILES/MIST_1.2_EEPtrk.h5'
        self.mistfile = mistfile
        with h5py.File(self.mistfile, "r") as misth5:
            self.make_lib(misth5, verbose=verbose)
        self.lib_as_grid()

        # Construct age weights.
        self._ageidx = self.predictions.index("loga")
        if ageweight:
            self.add_age_weights()

        # Construct grid.
        self.build_interpolator()

    def make_lib(self, misth5, verbose=True):
        """
        Convert the HDF5 input to ndarrays for labels and outputs. These
        are stored as `libparams` and `output` attributes, respectively.

        Parameters
        ----------
        misth5 : file
            Open hdf5 file to the MIST models.

        verbose : bool, optional
            Whether to print progress. Default is `True`.

        """

        if verbose:
            sys.stderr.write("Constructing MIST library...")
        cols = [rename[p] for p in self.labels]
        self.libparams = np.concatenate([np.array(misth5[z])[cols]
                                         for z in misth5["index"]])
        self.libparams.dtype.names = tuple(self.labels)

        cols = [rename[p] for p in self.predictions]
        try:
            # Assume all columns filled.
            self.output = [np.concatenate([misth5[z][p]
                                           for z in misth5["index"]])
                           for p in cols]
            self.output = np.array(self.output).T
        except:
            # If this fails, assume that [a/Fe] (afe_surf) is missing.
            # Substitute in for [Fe/H] (feh_surf) and then fill in with zeros.
            afe_surf_idx = np.where(rename["afe_surf"] == np.array(cols))[0][0]
            cols[afe_surf_idx] = rename["feh_surf"]
            self.output = [np.concatenate([misth5[z][p]
                                           for z in misth5["index"]])
                           for p in cols]
            self.output = np.array(self.output).T
            self.output[:, afe_surf_idx] *= 0.
            pass

        if verbose:
            sys.stderr.write("done!\n")

    def lib_as_grid(self):
        """
        Convert the library parameters to pixel indices in each dimension.
        The original grid points and binwidths are stored in `gridpoints`
        and `binwidths`, respectively, and the indices are stored as `X`.

        """

        # Get the unique gridpoints in each param
        self.gridpoints = {}
        self.binwidths = {}
        for p in self.labels:
            self.gridpoints[p] = np.unique(self.libparams[p])
            self.binwidths[p] = np.diff(self.gridpoints[p])

        # Digitize the library parameters
        X = np.array([np.digitize(self.libparams[p], bins=self.gridpoints[p],
                                  right=True) for p in self.labels])
        self.X = X.T

        self.mini_bound = self.gridpoints['mini'].min()

    def add_age_weights(self, verbose=True):
        """
        Compute the age gradient `d(age)/d(EEP)` over the EEP grid. Results
        are added to `output` and `predictions` so that the appropriate
        age weight is generated whenever a model is called.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress. Default is `True`.

        """

        # Check that we indeed have `loga` as a parameter.
        assert ("loga" in self.predictions)

        # Loop over tracks.
        age_ind = self._ageidx
        ageweights = np.zeros(len(self.libparams))
        for i, m in enumerate(self.gridpoints["mini"]):
            for j, z in enumerate(self.gridpoints["feh"]):
                for k, a in enumerate(self.gridpoints["afe"]):
                    if verbose:
                        sys.stderr.write("\rComputing age weights for track "
                                         "(mini, feh, afe) = "
                                         "({0}, {1}, {2})          "
                                         .format(m, z, a))
                        sys.stderr.flush()
                    # Get indices for this track.
                    inds = ((self.libparams["mini"] == m) &
                            (self.libparams["feh"] == z) &
                            (self.libparams["afe"] == a))
                    # Store delta(ages). Assumes tracks are ordered by age.
                    try:
                        agewts = np.gradient(10**self.output[inds, age_ind])
                        ageweights[inds] = agewts
                    except:
                        pass

        # Append results to outputs.
        self.output = np.hstack([self.output, ageweights[:, None]])
        self.predictions += ["agewt"]

        if verbose:
            sys.stderr.write('\n')

    def build_interpolator(self):
        """
        Construct the `~scipy.interpolate.RegularGridInterpolator` object
        used to generate fast predictions. The re-structured grid is stored
        under `grid_dims`, `xgrid`, and `ygrid`, while the interpolator object
        is stored under `interpolator`.

        """

        # Set up grid.
        self.grid_dims = np.append([len(self.gridpoints[p])
                                    for p in self.labels],
                                   self.output.shape[-1])
        self.xgrid = tuple([self.gridpoints[l] for l in self.labels])
        self.ygrid = np.zeros(self.grid_dims) + np.nan
        for x, y in zip(self.X, self.output):
            self.ygrid[tuple(x)] = y

        # Check for singular afe value.
        if self.grid_dims[-2] == 1:
            # Pad afe value.
            afe_val = self.xgrid[-1][0]
            xgrid = list(self.xgrid)
            xgrid[-1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)
            # Copy values over in predictions.
            self.grid_dims[-2] += 1
            ygrid = np.empty(self.grid_dims)
            ygrid[:, :, :, 0, :] = np.array(self.ygrid[:, :, :, 0, :])  # left
            ygrid[:, :, :, 1, :] = np.array(self.ygrid[:, :, :, 0, :])  # right
            self.ygrid = np.array(ygrid)

        # Initialize interpolator.
        self.interpolator = RegularGridInterpolator(self.xgrid, self.ygrid,
                                                    method='linear',
                                                    bounds_error=False,
                                                    fill_value=np.nan)

    def get_predictions(self, labels, apply_corr=True, corr_params=None):
        """
        Returns interpolated predictions for the input set of labels.

        Parameters
        ----------
        labels : 1-D or 2-D `~numpy.ndarray` of shape `(Nlabel, Nobj)`
            A set of labels we are interested in generating predictions for.

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective
            temperature and radius as a function of the input labels.
            Default is `True`.

        corr_params : tuple, optional
            Parameters that are used to generate the empirical corrections.
            If not provided, the default values are used.
            See `get_corrections` for additional details.

        Returns
        -------
        preds : 1-D or 2-D `~numpy.ndarray` of shape `(Npred, Nobj)`
            The set of predictions (1-D or 2-D) corresponding to the input
            `labels`.

        """

        labels = np.array(labels)
        ndim = labels.ndim
        if ndim == 1:
            preds = self.interpolator(labels)[0]
        elif ndim == 2:
            preds = self.interpolator(labels)
        else:
            raise ValueError("Input `labels` not 1-D or 2-D.")

        if apply_corr:
            corrs = self.get_corrections(labels, corr_params=corr_params)
            if ndim == 1:
                dlogt, dlogr = corrs
                preds[self.logt_idx] += dlogt
                preds[self.logl_idx] += 2. * dlogr
                preds[self.logg_idx] -= 2. * dlogr
            elif ndim == 2:
                dlogt, dlogr = corrs.T
                preds[:, self.logt_idx] += dlogt
                preds[:, self.logl_idx] += 2. * dlogr
                preds[:, self.logg_idx] -= 2. * dlogr

        return preds

    def get_corrections(self, labels, corr_params=None):
        """
        Returns interpolated corrections in some predictions for the input
        set of labels.

        Parameters
        ----------
        labels : 1-D or 2-D `~numpy.ndarray` of shape `(Nlabel, Nobj)`
            A set of labels we are interested in generating predictions for.

        corr_params : tuple, optional
            A tuple of `(dtdm, drdm, msto_smooth, feh_coef)` that are used to
            generate the empirical correction as a function of mass,
            metallicity, and EEP. Note that corrections are defined
            so that they do not affect predictions on the main sequence
            above 1 solar mass.
            `dtdm` adjusts the log(effective temperature) as a function
            of mass, while `drdm` adjusts the radius. `msto_smooth` sets the
            EEP scale for the exponential decay used to smoothly transition
            back to the underlying theoretical parameters
            around the Main Sequence Turn-Off (MSTO) point at `eep=454`.
            `feh_scale` is the coefficient used to enhance/suppress
            the magnitude of the effect as a function of metallicity
            following `np.exp(feh_scale * feh)`.
            If not provided, the default values of `dtdm=0.09`, `drdm=-0.09`,
            `msto_smooth = 30.`, and `feh_scale = 0.5` are used.

        Returns
        -------
        corrs : 1-D or 2-D `~numpy.ndarray` of shape `(Ncorr, Nobj)`
            The set of corrections (1-D or 2-D) corresponding to the input
            `labels`.

        """

        # Extract relevant parameters.
        labels = np.array(labels)
        ndim = labels.ndim
        mini, eep, feh = labels[[self.mini_idx, self.eep_idx, self.feh_idx]]
        if corr_params is not None:
            dtdm, drdm, msto_smooth, feh_scale = corr_params
        else:
            dtdm, drdm, msto_smooth, feh_scale = 0.09, -0.09, 30., 0.5

        # Compute logt and logr corrections.
        dlogt = np.log10(1. + (mini - 1.) * dtdm)  # Teff
        dlogr = np.log10(1. + (mini - 1.) * drdm)  # radius

        # EEP suppression
        ecorr = 1 - 1. / (1. + np.exp(-(eep - 454) / msto_smooth))

        # [Fe/H] dependence
        fcorr = np.exp(feh_scale * feh)

        # Combined effect.
        dlogt *= ecorr * fcorr
        dlogr *= ecorr * fcorr

        if ndim == 1:
            if mini >= 1.:
                corrs = np.array([0., 0.])
            else:
                corrs = np.array([dlogt, dlogr])
        elif ndim == 2:
            dlogt[mini >= 1.] = 0.
            dlogr[mini >= 1.] = 0.
            corrs = np.c_[dlogt, dlogr]
        else:
            raise ValueError("Input `labels` not 1-D or 2-D.")

        return corrs


class SEDmaker(MISTtracks):
    """
    An object that generates photometry interpolated from MIST tracks in
    EEP, initial mass, and metallicity using artificial neural networks.

    Parameters
    ----------
    filters : list of strings, optional
        The names of filters that photometry should be computed for. If not
        provided, photometry will be computed for all available filters.

    nnfile : str, optional
        The neural network file used to generate fast predictions.
        If not provided, this will default to `nnMIST_BC.h5` and is extracted
        from `data/DATAFILES/`.

    mistfile : str, optional
        The name of the HDF5 file containing the MIST tracks. Default is
        `MIST_1.2_EEPtrk.h5` and is extracted from the `data/DATAFILES/`.

    predictions : iterable of shape `(4)`, optional
        The names of the parameters to output at the request location in
        the `labels` parameter space. Default is
        `["loga", "logl", "logt", "logg"]`.
        **Do not modify this unless you know what you're doing.**

    ageweight : bool, optional
        Whether to compute the associated d(age)/d(EEP) weights at each
        EEP grid point, which are needed when applying priors in age.
        Default is `True`.

    verbose : bool, optional
        Whether to output progress to `~sys.stderr`. Default is `True`.

    """

    def __init__(self, filters=None, nnfile=None, mistfile=None,
                 predictions=["loga", "logl", "logt", "logg",
                              "feh_surf", "afe_surf"],
                 ageweight=True, verbose=True):

        # Initialize filters.
        if filters is None:
            filters = FILTERS
        self.filters = filters
        if verbose:
            sys.stderr.write('Filters: {}\n'.format(filters))

        # Initialize underlying MIST tracks.
        super(SEDmaker, self).__init__(mistfile=mistfile,
                                       predictions=predictions,
                                       ageweight=ageweight,
                                       verbose=verbose)

        # Initialize NNs.
        self.FNNP = FastNNPredictor(filters=filters, nnfile=nnfile,
                                    verbose=verbose)

    def get_sed(self, mini=1., eep=350., feh=0., afe=0., av=0., rv=3.3, smf=0.,
                dist=1000., loga_max=10.14, eep_binary_max=480.,
                tol=1e-3, mini_bound=0.5, apply_corr=True, corr_params=None,
                eep2=None, return_eep2=False, return_dict=True, **kwargs):
        """
        Generate and return the Spectral Energy Distribution (SED)
        and associated parameters for a given set of inputs.

        Parameters
        ----------
        mini : float, optional
            Initial mass in units of solar masses. Default is `1.`.

        eep : float, optional
            Equivalent evolutionary point (EEP). See the MIST documentation
            for additional details on how these are defined.
            Default is `350.`.

        feh : float, optional
            Metallicity defined logarithmically in terms of solar metallicity.
            Default is `0.`.

        afe : float, optional
            Alpha-enhancement defined logarithmically in terms of solar values.
            Default is `0.`.

        av : float, optional
            Dust attenuation defined in terms of reddened V-band magnitudes.
            Default is `0.`.

        rv : float, optional
            Change in the reddening vector in terms of R(V)=A(V)/E(B-V).
            Default is `3.3`.

        smf : float, optional
            Secondary mass fraction for unresolved binary. Default is `0.`
            (i.e. single stellar system).

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        eep_binary_max : float, optional
            The maximum EEP where binaries are permitted. By default, binaries
            are disallowed once the primary begins its giant expansion
            after turning off the Main Sequence at `eep=480.`.

        tol : float, optional
            The tolerance in the `loga` solution for a given EEP. Used when
            computing the secondary in unresolved binaries.
            Default is `1e-3`.

        mini_bound : float, optional
            A hard bound on the initial mass. Any models below this threshold
            will be masked (including those from binary components).
            Default is `0.5`.

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective
            temperature and radius as a function of mass, metallicity, and EEP.
            Default is `True`.

        corr_params : tuple, optional
            Parameters that are used to generate the empirical corrections.
            If not provided, the default values are used.
            See `get_corrections` for additional details.

        eep2 : float, optional
            Equivalent evolutionary point (EEP) of the secondary. If not
            provided, this will be solved for using `get_eep`.

        return_eep2 : float, optional
            Whether to return the EEP of the secondary. Default is `False`.

        return_dict : bool, optional
            Whether to return the parameters as a dictionary.
            Default is `True`.

        Returns
        -------
        sed : `~numpy.ndarray` of shape `(Nfilters,)`
            The predicted SED in magnitudes in the initialized filters.

        params : dict or array of length `(Npred,)`
            The corresponding predicted parameters associated with the given
            SED of the primary component.

        params2 : dict or array of length `(Npred,)`
            The corresponding predicted parameters associated with the given
            SED of the secondary component.

        """

        # Grab input labels.
        labels = {'mini': mini, 'eep': eep, 'feh': feh, 'afe': afe}
        labels = np.array([labels[l] for l in self.labels])  # reorder

        # Generate predictions.
        params_arr = self.get_predictions(labels, apply_corr=apply_corr,
                                          corr_params=corr_params)
        params = dict(zip(self.predictions, params_arr))  # convert to dict
        sed = np.full(self.FNNP.NFILT, np.nan)  # SED

        # Binary predictions.
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.predictions, params_arr2))

        # Generate SED.
        mini_min = max(self.mini_bound, mini_bound)
        loga = params['loga']
        if loga <= loga_max:
            # Compute SED.
            sed = self.FNNP.sed(logl=params["logl"], logt=params["logt"],
                                logg=params["logg"],
                                feh_surf=params["feh_surf"],
                                afe=params["afe_surf"],
                                av=av, rv=rv, dist=dist)
            # Add in unresolved binary component if we're on the Main Sequence.
            if smf > 0. and eep <= eep_binary_max and mini * smf >= mini_min:
                # Generate predictions for secondary binary component.
                if eep2 is None:
                    # Convert loga to EEP.
                    eep2 = self.get_eep(loga, mini=mini, eep=eep,
                                        feh=feh, smf=smf, tol=tol)
                labels2 = {'mini': mini * smf, 'eep': eep2,
                           'feh': feh, 'afe': afe}
                labels2 = np.array([labels2[l] for l in self.labels])
                params_arr2 = self.get_predictions(labels2,
                                                   apply_corr=apply_corr,
                                                   corr_params=corr_params)
                params2 = dict(zip(self.predictions, params_arr2))
                # Compute SED.
                sed2 = self.FNNP.sed(logl=params2["logl"],
                                     logt=params2["logt"],
                                     logg=params2["logg"],
                                     feh_surf=params2["feh_surf"],
                                     afe=params2["afe_surf"],
                                     av=av, rv=rv, dist=dist)
                # Combine primary and secondary components.
                sed = add_mag(sed, sed2)
            elif smf > 0.:
                # Overwrite original prediction with "empty" SED.
                sed = np.full(self.FNNP.NFILT, np.nan)

        # If we are not returning a dictionary, overwrite `params`.
        if not return_dict:
            params, params2 = params_arr, params_arr2

        if return_eep2:
            return sed, params, params2, eep2
        else:
            return sed, params, params2

    def get_eep(self, loga, mini=1., eep=350., feh=0., afe=0., smf=1.,
                tol=1e-3):
        """
        Compute the corresponding EEP for a particular age.

        Parameters
        ----------
        loga : float
            The base-10 logarithm of the age.

        mini : float, optional
            Initial mass in units of solar masses. Default is `1.`.

        eep : float, optional
            Equivalent evolutionary point (EEP) used as an initial guess.
            See the MIST documentation for additional details on how
            these are defined. Default is `350.`.

        feh : float, optional
            Metallicity defined logarithmically in terms of solar metallicity.
            Default is `0.`.

        afe : float, optional
            Alpha-enhancement defined logarithmically in terms of solar values.
            Default is `0.`.

        smf : float, optional
            Secondary mass fraction for unresolved binary. Default is `1.`
            (equal-mass binary stellar system).

        tol : float, optional
            The tolerance in the `loga` solution for a given EEP.
            Default is `1e-3`.

        """

        aidx = self._ageidx  # index of age variable

        # Define loss function.
        def loss(x):
            loga_pred = self.get_predictions([mini * smf, x, feh, afe])[aidx]
            return (loga_pred - loga)**2
        # Find best-fit age that minimizes loss.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            res = minimize(loss, eep)
        # Check against tolerance.
        if res['fun'] < tol:
            eep2 = res['x'][0]
        else:
            eep2 = np.nan

        return eep2

    def make_grid(self, mini_grid=None, eep_grid=None, feh_grid=None,
                  afe_grid=None, smf_grid=None, av_grid=None, av_wt=None,
                  rv_grid=None, rv_wt=None, dist=1000., loga_max=10.14,
                  eep_binary_max=480., mini_bound=0.5,
                  apply_corr=True, corr_params=None, verbose=True, **kwargs):
        """
        Generate and return SED predictions over a grid in inputs.
        Reddened photometry is generated by fitting
        a linear relationship in Av and Rv over the
        specified (weighted) Av and Rv grids, whose coefficients are stored.

        Parameters
        ----------
        mini_grid : `~numpy.ndarray`, optional
            Grid in initial mass (in units of solar masses). If not provided,
            the default is a grid from 0.5 to 2.0 with a resolution of 0.025.

        eep_grid : `~numpy.ndarray`, optional
            Grid in EEP. If not provided, the default is an adaptive grid with:
            (1) resolution of 12 from 202 to 454 (on the Main Sequence) and
            (2) resolution of 3 from 454 to 808 (off the Main Sequence).

        feh_grid : `~numpy.ndarray`, optional
            Grid in metallicity (defined logarithmically in units of solar
            metallicity). If not provided, the default is an adaptive
            grid with:
            (1) resolution of 0.1 from -3.0 to -2.0, and
            (3) resolution of 0.05 from -2.0 to +0.05.

        afe_grid : `~numpy.ndarray`, optional
            Grid in alpha-enhancement (defined logarithmically in units of
            solar values). If not provided, the default is a grid from
            -0.2 to +0.6 with a resolution of 0.2.

        smf_grid : `~numpy.ndarray`, optional
            Grid in secondary mass fraction from `[0., 1.]` for computing
            unresolved binaries. If not provided, the default is `[0.]`
            (single star only).

        av_grid : `~numpy.ndarray`, optional
            Grid in dust attenuation defined in terms of reddened V-band
            magnitudes. Used to fit for a linear "reddening vector".
            If not provided, the default is a grid from 0. to 1.5
            with a resolution of 0.3.

        av_wt : `~numpy.ndarray`, optional
            The associated weights over the provided `av_grid` to be used when
            fitting. If not provided, the default is `(1e-5 + av_grid)**-1`.
            This forces the fit to go through `Av=0.` with 1/x weighting
            for the remaining points.

        rv_grid : `~numpy.ndarray`, optional
            Grid in differential dust attenuation defined in terms of R(V).
            Used to fit for a linear "differential reddening vector".
            If not provided, the default is a grid from 2.4 to 4.2
            with a resolution of 0.3.

        rv_wt : `~numpy.ndarray`, optional
            The associated weights over the provided `rv_grid` to be used when
            fitting. If not provided, the default is
            `np.exp(-np.abs(rv_grid - 3.3) / 0.5)`.

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        eep_binary_max : float, optional
            The maximum EEP where binaries are permitted. By default, binaries
            are disallowed once the primary begins its giant expansion
            after turning off the Main Sequence at `eep=480.`.

        mini_bound : float, optional
            A hard bound on the initial mass. Any models below this threshold
            will be masked (including those from binary components).
            Default is `0.5`.

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective
            temperature and radius as a function of mass, metallicity, and EEP.
            Default is `True`.

        corr_params : tuple, optional
            Parameters that are used to generate the empirical corrections.
            If not provided, the default values are used.
            See `get_corrections` for additional details.

        verbose : bool, optional
            Whether to print progress. Default is `True`.

        """

        # Initialize grid.
        labels = ['mini', 'eep', 'feh', 'afe', 'smf']
        ltype = np.dtype([(n, np.float) for n in labels])
        if mini_grid is None:  # initial mass
            mini_grid = np.arange(0.5, 2.0 + 1e-5, 0.025)
        if eep_grid is None:  # EEP
            eep_grid = np.concatenate([np.arange(202., 454., 6.),
                                       np.arange(454., 808. + 1e-5, 2.)])
        if feh_grid is None:  # metallicity
            feh_grid = np.concatenate([np.arange(-3., -2., 0.1),
                                       np.arange(-2., 0.5 + 1e-5, 0.05)])
        if afe_grid is None:  # alpha-enhancement
            afe_grid = np.arange(-0.2, 0.6 + 1e-5, 0.2)
        if smf_grid is None:  # binary secondary mass fraction
            smf_grid = np.array([0.])
        if av_grid is None:  # reddening
            av_grid = np.arange(0., 1.5 + 1e-5, 0.3)
            av_grid[-1] -= 1e-5
        if av_wt is None:  # Av weights
            # Pivot around Av=0 point with inverse Av weighting.
            av_wt = (1e-5 + av_grid)**-1.
        if rv_grid is None:  # differential reddening
            rv_grid = np.arange(2.4, 4.2 + 1e-5, 0.3)
        if av_wt is None:  # Rv weights
            # Exponential weighting with width of dRv=0.5.
            rv_wt = np.exp(-np.abs(rv_grid - 3.3) / 0.5)

        # Create grid.
        self.grid_label = np.array(list(product(*[mini_grid, eep_grid,
                                                  feh_grid, afe_grid,
                                                  smf_grid])),
                                   dtype=ltype)
        Ngrid = len(self.grid_label)

        # Generate SEDs on the grid.
        ptype = np.dtype([(n, np.float) for n in self.predictions])
        stype = np.dtype([(n, np.float, 3) for n in self.filters])
        self.grid_sed = np.full(Ngrid, np.nan, dtype=stype)
        self.grid_param = np.full(Ngrid, np.nan, dtype=ptype)
        self.grid_sel = np.ones(Ngrid, dtype='bool')

        percentage = -99
        ttot, t1 = 0., time.time()
        for i, (mini, eep, feh, afe, smf) in enumerate(self.grid_label):

            # Compute model and parameter predictions.
            (sed, params,
             params2, eep2) = self.get_sed(mini=mini, eep=eep, feh=feh,
                                           afe=afe, smf=smf, av=0., rv=3.3,
                                           dist=dist, loga_max=loga_max,
                                           eep_binary_max=eep_binary_max,
                                           return_dict=False, return_eep2=True,
                                           apply_corr=apply_corr,
                                           corr_params=corr_params,
                                           mini_bound=mini_bound)
            # Save predictions for primary.
            self.grid_param[i] = tuple(params)

            # Deal with non-existent SEDS.
            if np.any(np.isnan(sed)) or np.any(np.isnan(params)):
                # Flag results and fill with `nan`s.
                self.grid_sel[i] = False
                self.grid_sed[i] = tuple(np.full((self.FNNP.NFILT, 1),
                                                 np.nan))
            else:
                # Compute fits for reddening.
                seds = np.array([[self.get_sed(mini=mini, eep=eep, feh=feh,
                                               afe=afe, smf=smf, eep2=eep2,
                                               av=av, rv=rv,
                                               dist=dist, loga_max=loga_max,
                                               eep_binary_max=eep_binary_max,
                                               return_dict=False,
                                               apply_corr=apply_corr,
                                               corr_params=corr_params,
                                               mini_bound=mini_bound)[0]
                                  for av in av_grid]
                                 for rv in rv_grid])
                sfits = np.array([polyfit(av_grid, s, 1, w=av_wt).T
                                  for s in seds])  # Av at fixed Rv
                sedr, seda = polyfit(rv_grid, sfits[:, :, 0], 1,
                                     w=rv_wt)  # Rv vector, Av vector
                self.grid_sed[i] = tuple(np.c_[sed, seda, sedr])

            # Get runtime.
            t2 = time.time()
            dt = t2 - t1
            ttot += dt
            tavg = ttot / (i + 1)
            test = tavg * (Ngrid - i - 1)
            t1 = t2

            # Print progress.
            new_percentage = int((i + 1) / Ngrid * 1e5)
            if verbose and new_percentage != percentage:
                percentage = new_percentage
                sys.stderr.write('\rConstructing grid {:6.3f}% ({:d}/{:d}) '
                                 '[mini={:6.3f}, eep={:6.3f}, feh={:6.3f}, '
                                 'afe={:6.3f}, smf={:6.3f}] '
                                 '(t/obj: {:3.3f} ms, '
                                 'est. remaining: {:10.3f} s)          '
                                 .format(percentage / 1.0e3, i + 1, Ngrid,
                                         mini, eep, feh, afe, smf,
                                         tavg * 1e3, test))
                sys.stderr.flush()

        if verbose:
            sys.stderr.write('\n')


class FastNN(object):
    """
    Object that wraps the underlying neural networks used to interpolate
    between grid points on the bolometric correction tables.

    Parameters
    ----------
    filters : list of strings, optional
        The names of filters that photometry should be computed for.
        If not provided, all available filters will be used.

    nnfile : str, optional
        The neutral network file. Default is `nnMIST_BC.h5` stored under
        `data/DATAFILES/`.

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    """

    def __init__(self, filters=None, nnfile=None, verbose=True):

        # Initialize values.
        if filters is None:
            filters = np.array(FILTERS)
        if nnfile is None:
            nnfile = 'data/DATAFILES/nnMIST_BC.h5'

        # Read in NN data.
        if verbose:
            sys.stderr.write('Initializing FastNN predictor...')
        self._load_NN(filters, nnfile)
        if verbose:
            sys.stderr.write('done!\n')

    def _load_NN(self, filters, nnfile):
        """
        Load in weight and bias from each neural network layer.

        """

        with h5py.File(nnfile, "r") as f:
            # Store weights and bias.
            self.w1 = np.array([f[fltr]['w1'] for fltr in filters])
            self.b1 = np.array([f[fltr]['b1'] for fltr in filters])
            self.w2 = np.array([f[fltr]['w2'] for fltr in filters])
            self.b2 = np.array([f[fltr]['b2'] for fltr in filters])
            self.w3 = np.array([f[fltr]['w3'] for fltr in filters])
            self.b3 = np.array([f[fltr]['b3'] for fltr in filters])
            xmin = np.array([f[fltr]['xmin'] for fltr in filters])
            xmax = np.array([f[fltr]['xmax'] for fltr in filters])
            if len(np.unique(xmin)) == 6 and len(np.unique(xmax)) == 6:
                self.xmin = xmin[0]
                self.xmax = xmax[0]
                self.xspan = (self.xmax - self.xmin)
            else:
                raise ValueError("Some of the neural networks have different "
                                 "`xmin` and `xmax` ranges for parameters.")

    def encode(self, x):
        """
        Rescale the `x` iterable.

        Parameters
        ----------
        x : `~numpy.ndarray` of shape `(Ninput,)`
            Input labels.

        Returns
        -------
        xp : `~numpy.ndarray` of shape `(Npred, 1)`
            Output predictions.

        """

        try:
            xp = (np.atleast_2d(x) - self.xmin[None, :]) / self.xspan[None, :]
            return xp.T
        except:
            xp = (np.atleast_2d(x) - self.xmin[:, None]) / self.xspan[:, None]
            return xp

    def sigmoid(self, a):
        """
        Transformed `a` via the sigmoid function.

        Parameters
        ----------
        a : `~numpy.ndarray` of shape `(Ninput,)`
            Input array.

        Returns
        -------
        a_t : `~numpy.ndarray` of shape `(Ninput)`
            Values after applying the sigmoid transform.

        """

        return 1. / (1. + np.exp(-a))

    def nneval(self, x):
        """
        Evaluate the neural network at the value of `x`.

        Parameters
        ----------
        x : `~numpy.ndarray` of shape `(Ninput,)`
            Input labels.

        """

        a1 = self.sigmoid(np.matmul(self.w1, self.encode(x)) + self.b1)
        a2 = self.sigmoid(np.matmul(self.w2, a1) + self.b2)
        y = np.matmul(self.w3, a2) + self.b3

        return np.squeeze(y)


class FastNNPredictor(FastNN):
    """
    Object that generates SED predictions for a provided set of filters
    using neural networks trained on bolometric correction tables.

    Parameters
    ----------
    filters : list of strings, optional
        The names of filters that photometry should be computed for.
        If not provided, all available filters will be used.

    nnfile : str, optional
        The neutral network file. Default is `nnMIST_BC.h5` stored under
        `data/DATAFILES/`.

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    """

    def __init__(self, filters=None, nnfile=None, verbose=True):

        # Initialize values.
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters
        self.NFILT = len(filters)
        if nnfile is None:
            nnfile = 'data/DATAFILES/nnMIST_BC.h5'
        super(FastNNPredictor, self).__init__(filters=filters, nnfile=nnfile,
                                              verbose=verbose)

    def sed(self, logt=3.8, logg=4.4, feh_surf=0., logl=0., afe=0.,
            av=0., rv=3.3, dist=1000., filt_idxs=slice(None)):
        """
        Returns the SED predicted by neural networks for the input set of
        physical parameters for a specified subset of bands. Predictions
        are in apparent magnitudes at the specified distance. See
        `filters` for the order of the bands.

        Parameters
        ----------
        logt : float, optional
            The base-10 logarithm of the effective temperature in Kelvin.
            Default is `3.8`.

        logg : float, optional
            The base-10 logarithm of the surface gravity in cgs units (cm/s^2).
            Default is `4.4`.

        feh_surf : float, optional
            The surface metallicity in logarithmic units of solar metallicity.
            Default is `0.`.

        logl : float, optional
            The base-10 logarithm of the luminosity in solar luminosities.
            Default is `0.`.

        afe : float, optional
            The alpha-enhancement in logarithmic units relative to solar
            values. Default is `0.`.

        av : float, optional
            Dust attenuation in units of V-band reddened magnitudes.
            Default is `0.`.

        rv : float, optional
            Change in the reddening vector in terms of R(V)=A(V)/E(B-V).
            Default is `3.3`.

        dist : float, optional
            Distance in parsecs. Default is `1000.`

        filt_idxs : iterable of shape `(Nfilt)`, optional
            Susbset of filter indices. If not provided, predictions in all
            filters will be returned.

        Returns
        -------
        sed : `~numpy.ndarray` of shape (Nfilt)
            Predicted SED in magnitudes.

        """

        # Compute distance modulus.
        mu = 5. * np.log10(dist) - 5.

        # Compute apparent magnitudes.
        x = np.array([10.**logt, logg, feh_surf, afe, av, rv])
        if np.all(np.isfinite(x)) and np.all((x >= self.xmin) &
                                             (x <= self.xmax)):
            # Check whether we're within the bounds of the neural net.
            # If we're good, compute the bolometric correction and convert
            # to apparent magnitudes.
            BC = self.nneval(x)
            m = -2.5 * logl + 4.74 - BC + mu
        else:
            # If we're out of bounds, return `np.nan` values.
            m = np.full(self.NFILT, np.nan)

        return np.atleast_1d(m)[filt_idxs]


class Isochrone(object):
    """
    An object that generates photometry interpolated from MIST isochrones in
    EEP, metallicity, and log(age) using artificial neural networks.

    Parameters
    ----------
    filters : list of strings, optional
        The names of filters that photometry should be computed for. If not
        provided, photometry will be computed for all available filters.

    nnfile : str, optional
        The neural network file used to generate fast predictions.
        If not provided, this will default to `nnMIST_BC.h5` and is extracted
        from `data/DATAFILES/`.

    mistfile : str, optional
        The name of the HDF5 file containing the MIST tracks. Default is
        `MIST_1.2_iso.h5` and is extracted from the `data/DATAFILES/`.

    predictions : iterable of shape `(4)`, optional
        The names of the parameters to output at the request location in
        the `labels` parameter space. Default is
        `["mini", "mass", "logl", "logt", "logr", "logg",`
        ` "feh_surf", "afe_surf"]`.
        **Do not modify this unless you know what you're doing.**

    verbose : bool, optional
        Whether to output progress to `~sys.stderr`. Default is `True`.

    """

    def __init__(self, filters=None, nnfile=None, mistfile=None,
                 predictions=None, verbose=True):

        # Initialize values.
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters
        if verbose:
            sys.stderr.write('Filters: {}\n'.format(filters))
        if nnfile is None:
            nnfile = 'data/DATAFILES/nnMIST_BC.h5'
        if mistfile is None:
            mistfile = 'data/DATAFILES/MIST_1.2_iso_vvcrit0.0.h5'
        if predictions is None:
            predictions = ["mini", "mass", "logl", "logt",
                           "logr", "logg", "feh_surf", "afe_surf"]
        self.predictions = predictions

        if verbose:
            sys.stderr.write('Constructing MIST isochrones...')

        # Load file.
        with h5py.File(mistfile, "r") as f:
            self.feh_grid = f['feh'][:]
            self.afe_grid = f['afe'][:]
            self.loga_grid = f['loga'][:]
            self.eep_grid = f['eep'][:]
            self.pred_grid = f['predictions'][:]
            self.pred_labels = f['predictions'].attrs['labels']

        # Initialize interpolator.
        self.build_interpolator()

        if verbose:
            sys.stderr.write('done!\n')

        # Initialize NNs.
        self.FNNP = FastNNPredictor(filters=filters, nnfile=nnfile,
                                    verbose=verbose)

    def build_interpolator(self):
        """
        Construct the `~scipy.interpolate.RegularGridInterpolator` object
        used to generate isochrones. The re-structured grid is stored under
        `grid_dims`, `xgrid`, and `ygrid`, while the interpolator object
        is stored under `interpolator`.

        """

        # Set up grid.
        self.feh_u = np.unique(self.feh_grid)
        self.afe_u = np.unique(self.afe_grid)
        self.loga_u = np.unique(self.loga_grid)
        self.eep_u = np.unique(self.eep_grid)
        self.xgrid = (self.feh_u, self.afe_u, self.loga_u, self.eep_u)
        self.grid_dims = np.array([len(self.xgrid[0]), len(self.xgrid[1]),
                                   len(self.xgrid[2]), len(self.xgrid[3]),
                                   len(self.pred_labels)], dtype='int')

        # Fill in "holes" if possible.
        for i in range(len(self.feh_u)):
            for j in range(len(self.afe_u)):
                for k in range(len(self.loga_u)):
                    # Select values where predictions exist.
                    sel = np.all(np.isfinite(self.pred_grid[i, j, k]), axis=1)
                    try:
                        # Linearly interpolate over built-in EEP grid.
                        pnew = [np.interp(self.eep_u, self.eep_u[sel], par,
                                          left=np.nan, right=np.nan)
                                for par in self.pred_grid[i, j, k, sel].T]
                        pnew = np.array(pnew).T  # copy and transpose
                        self.pred_grid[i, j, k] = pnew  # assign predictions
                    except:
                        # Fail silently and give up.
                        pass

        # Check for singular afe value.
        if self.grid_dims[1] == 1:
            # Pad afe value.
            afe_val = self.xgrid[1][0]
            xgrid = list(self.xgrid)
            xgrid[1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)
            # Copy values over in predictions.
            self.grid_dims[1] += 1
            ygrid = np.empty(self.grid_dims)
            ygrid[:, 0, :, :, :] = np.array(self.pred_grid[:, 0, :, :, :])
            ygrid[:, 1, :, :, :] = np.array(self.pred_grid[:, 0, :, :, :])
            self.pred_grid = np.array(ygrid)

        # Initialize interpolator
        self.interpolator = RegularGridInterpolator(self.xgrid,
                                                    self.pred_grid,
                                                    method='linear',
                                                    bounds_error=False,
                                                    fill_value=np.nan)

        # Set label references.
        self.logt_idx = np.where(np.array(self.predictions) == 'logt')[0][0]
        self.logl_idx = np.where(np.array(self.predictions) == 'logl')[0][0]
        self.logg_idx = np.where(np.array(self.predictions) == 'logg')[0][0]
        self.feh_surf_idx = np.where(np.array(self.predictions) ==
                                     'feh_surf')[0][0]
        self.mini_idx = np.where(np.array(self.predictions) == 'mini')[0][0]

    def get_predictions(self, feh=0., afe=0., loga=8.5, eep=None,
                        apply_corr=True, corr_params=None):
        """
        Returns physical predictions for a given metallicity, log(age), and
        EEP (either a single value or a grid).

        Parameters
        ----------
        feh : float, optional
            Metallicity defined logarithmically in terms of solar metallicity.
            Default is `0.`.

        afe : float, optional
            Alpha-enhancement defined logarithmically in terms of solar values.
            Default is `0.`.

        loga : float, optional
            Log(age) where age is in years. Default is `8.5`.

        eep : `~numpy.ndarray` of shape `(Neep)`, optional
            Equivalent evolutionary point(s) (EEPs). See the MIST documentation
            for additional details on how these are defined. If not provided,
            the default EEP grid defined on initialization will be used.

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective
            temperature and radius as a function of mass, metallicity, and EEP.
            Default is `True`.

        corr_params : tuple, optional
            Parameters that are used to generate the empirical corrections.
            If not provided, the default values are used.
            See `get_corrections` for additional details.

        Returns
        -------
        preds : 2-D `~numpy.ndarray` of shape `(Npred, Nobj)`
            The set of predictions corresponding to the input
            `eep` values.

        """

        # Fill out input labels.
        if eep is None:
            eep = self.eep_u
        eep = np.array(eep, dtype='float')
        feh = np.full_like(eep, feh)
        afe = np.full_like(eep, afe)
        loga = np.full_like(eep, loga)
        labels = np.c_[feh, afe, loga, eep]

        # Generate predictions.
        preds = self.interpolator(labels)
        mini = preds[:, self.mini_idx]

        # Apply corrections.
        if apply_corr:
            corrs = self.get_corrections(mini=mini, feh=feh, eep=eep,
                                         corr_params=corr_params)
            dlogt, dlogr = corrs.T
            preds[:, self.logt_idx] += dlogt
            preds[:, self.logl_idx] += 2. * dlogr
            preds[:, self.logg_idx] -= 2. * dlogr

        return preds

    def get_corrections(self, mini=1., feh=0., eep=350., corr_params=None):
        """
        Returns interpolated corrections in some predictions for the input
        set of labels.

        Parameters
        ----------
        mini : float or `~numpy.ndarray`, optional
            Initial mass in units of solar masses. Default is `1.`.

        feh : float or `~numpy.ndarray`, optional
            Metallicity defined logarithmically in terms of solar metallicity.
            Default is `0.`.

        eep : float or `~numpy.ndarray`, optional
            Equivalent evolutionary point(s) (EEPs). See the MIST documentation
            for additional details on how these are defined. Default is `350.`.

        corr_params : tuple, optional
            A tuple of `(dtdm, drdm, msto_smooth, feh_coef)` that are used to
            generate the empirical correction as a function of mass,
            metallicity, and EEP. Note that corrections are defined
            so that they do not affect predictions on the main sequence
            above 1 solar mass.
            `dtdm` adjusts the log(effective temperature) as a function
            of mass, while `drdm` adjusts the radius. `msto_smooth` sets the
            EEP scale for the exponential decay used to smoothly transition
            back to the underlying theoretical parameters
            around the Main Sequence Turn-Off (MSTO) point at `eep=454`.
            `feh_scale` is the coefficient used to enhance/suppress
            the magnitude of the effect as a function of metallicity
            following `np.exp(feh_scale * feh)`.
            If not provided, the default values of `dtdm=0.09`, `drdm=-0.09`,
            `msto_smooth = 30.`, and `feh_scale = 0.5` are used.

        Returns
        -------
        corrs : 1-D or 2-D `~numpy.ndarray` of shape `(Ncorr, Nobj)`
            The set of corrections (1-D or 2-D) corresponding to the input
            `labels`.

        """

        if corr_params is not None:
            dtdm, drdm, msto_smooth, feh_scale = corr_params
        else:
            dtdm, drdm, msto_smooth, feh_scale = 0.09, -0.09, 30., 0.5

        # Baseline corrections to logt and logr.
        dlogt = np.log10(1. + (mini - 1.) * dtdm)  # Teff
        dlogr = np.log10(1. + (mini - 1.) * drdm)  # radius

        # EEP suppression.
        ecorr = 1 - 1. / (1. + np.exp(-(eep - 454) / msto_smooth))

        # [Fe/H] dependence.
        fcorr = np.exp(feh_scale * feh)

        # Combined effect.
        dlogt *= ecorr * fcorr
        dlogr *= ecorr * fcorr

        # Remove corrections above 1 solar mass.
        labels = np.c_[mini, eep, feh]
        if labels.shape[0] == 1:
            if mini >= 1.:
                corrs = np.array([0., 0.])
            else:
                corrs = np.array([dlogt, dlogr])
        else:
            dlogt[mini >= 1.] = 0.
            dlogr[mini >= 1.] = 0.
            corrs = np.c_[dlogt, dlogr]

        return corrs

    def get_seds(self, feh=0., afe=0., loga=8.5, eep=None,
                 av=0., rv=3.3, smf=0., dist=1000.,
                 mini_bound=0.5, eep_binary_max=480.,
                 apply_corr=True, corr_params=None,
                 return_dict=True, **kwargs):
        """
        Generate and return the Spectral Energy Distribution (SED)
        and associated parameters for a given set of inputs.

        Parameters
        ----------
        feh : float, optional
            Metallicity defined logarithmically in terms of solar metallicity.
            Default is `0.`.

        afe : float, optional
            Alpha-enhancement defined logarithmically in terms of solar values.
            Default is `0.`.

        loga : float, optional
            Log(age) where age is in years. Default is `8.5`.

        eep : `~numpy.ndarray` of shape `(Neep)`, optional
            Equivalent evolutionary point(s) (EEPs). See the MIST documentation
            for additional details on how these are defined. If not provided,
            the default EEP grid defined on initialization will be used.

        av : float, optional
            Dust attenuation defined in terms of reddened V-band magnitudes.
            Default is `0.`.

        rv : float, optional
            Change in the reddening vector in terms of R(V)=A(V)/E(B-V).
            Default is `3.3`.

        smf : float, optional
            Secondary mass fraction for unresolved binary. Default is `0.`
            (single stellar system).

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        mini_bound : float, optional
            A hard bound on the initial mass. Any models below this threshold
            will be masked (including those from binary components).
            Default is `0.5`.

        eep_binary_max : float, optional
            The maximum EEP where binaries are permitted. By default, binaries
            are disallowed once the primary begins its giant expansion
            after turning off the Main Sequence at `eep=480.`.

        apply_corr : bool, optional
            Whether to apply empirical corrections to the effective
            temperature and radius as a function of mass, metallicity, and EEP.
            Default is `True`.

        corr_params : tuple, optional
            Parameters that are used to generate the empirical corrections.
            If not provided, the default values are used.
            See `get_corrections` for additional details.

        return_dict : bool, optional
            Whether to return the parameters as a dictionary.
            Default is `True`.

        Returns
        -------
        sed : `~numpy.ndarray` of shape `(Nfilters,)`
            The predicted SED in magnitudes in the initialized filters.

        params : dict or array of length `(Npred,)`
            The corresponding predicted parameters associated with the given
            SED of the primary component.

        params2 : dict or array of length `(Npred,)`
            The corresponding predicted parameters associated with the given
            SED of the secondary component.

        """

        # Initialize EEPs.
        if eep is None:
            eep = self.eep_u
        eep = np.array(eep, dtype='float')
        Neep = len(eep)

        # Generate predictions.
        params_arr = self.get_predictions(feh=feh, afe=afe, loga=loga, eep=eep,
                                          apply_corr=apply_corr,
                                          corr_params=corr_params)
        params = dict(zip(self.predictions, params_arr.T))  # convert to dict

        # Generate isochrone SEDs.
        seds = np.full((Neep, self.FNNP.NFILT), np.nan)
        for i in range(Neep):
            if params["mini"][i] >= mini_bound:
                seds[i] = self.FNNP.sed(logl=params["logl"][i],
                                        logt=params["logt"][i],
                                        logg=params["logg"][i],
                                        feh_surf=params["feh_surf"][i],
                                        afe=params["afe_surf"][i],
                                        av=av, rv=rv, dist=dist)

        # Add in binaries (if appropriate).
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.predictions, params_arr2.T))
        if 0. < smf < 1.:
            mini = params["mini"]
            mini2 = mini * smf
            mini_mask = np.where(np.isfinite(mini))[0]
            if len(mini_mask) > 0:
                eep2 = np.interp(mini2, mini[mini_mask], eep[mini_mask],
                                 left=np.nan, right=np.nan)
            else:
                eep2 = np.full_like(eep, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore bad values
                eep2[(eep2 > eep_binary_max) | (eep > eep_binary_max)] = np.nan
            params_arr2 = self.get_predictions(feh=feh, afe=afe,
                                               loga=loga, eep=eep2,
                                               apply_corr=apply_corr,
                                               corr_params=corr_params)
            params2 = dict(zip(self.predictions, params_arr2.T))
            seds2 = np.full((Neep, self.FNNP.NFILT), np.nan)
            for i in range(Neep):
                if params2["mini"][i] >= mini_bound:
                    seds2[i] = self.FNNP.sed(logl=params2["logl"][i],
                                             logt=params2["logt"][i],
                                             logg=params2["logg"][i],
                                             feh_surf=params2["feh_surf"][i],
                                             afe=params2["afe_surf"][i],
                                             av=av, rv=rv, dist=dist)
            seds = add_mag(seds, seds2)
        elif smf == 1.:
            seds[eep <= eep_binary_max] -= 2.5 * np.log10(2.)
            params2, params_arr2 = deepcopy(params), deepcopy(params_arr2)

        # If we are not returning a dictionary, overwrite `params`.
        if not return_dict:
            params, params2 = params_arr, params_arr2

        return seds, params, params2
