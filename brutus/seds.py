#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collection of methods that generate the underlying grid of SED models. Code
contributed by Ben Johnson.

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
from copy import deepcopy
from itertools import product
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy import polyfit

import minesweeper
from minesweeper.photANN import ANN

from .filters import FILTERS

# Rename parameters from what is in the MIST HDF5 file.
# This makes it easier to use parameter names as keyword arguments.
rename = {"mini": "initial_mass",  # input parameters
          "eep": "EEP",
          "feh": "initial_[Fe/H]",
          "afe": "initial_[a/Fe]",
          "mass": "star_mass",  # outputs
          "feh_surf": "[Fe/H]",
          "loga": "log_age",
          "logt": "log_Teff",
          "logg": "log_g",
          "logl": "log_L",
          "logr": "log_R"}

__all__ = ["MISTtracks", "SEDmaker", "FastNN", "FastPaynePredictor"]


class MISTtracks(object):
    """
    An object that linearly interpolates the MIST tracks in EEP, initial mass,
    and metallicity. Uses `~scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    mistfile : str, optional
        The name of the HDF5 file containing the MIST tracks. Default is
        `MIST_1.2_EEPtrk.h5` and is extracted from the `minesweeper`
        home path.

    labels : iterable of shape `(3)`, optional
        The names of the parameters on which to interpolate. This defaults to
        `["mini", "eep", "feh"]`. **Change this only if you know what**
        **you're doing.**

    predictions : iterable of shape `(4)`, optional
        The names of the parameters to output at the request location in
        the `labels` parameter space. Default is
        `["loga", "logl", "logt", "logg"]`.

    ageweight : bool, optional
        Whether to compute the associated d(age)/d(EEP) weights at each
        EEP grid point, which are needed when applying priors in age.
        Default is `True`.

    verbose : bool, optional
        Whether to output progress to `~sys.stderr`. Default is `True`.

    """

    def __init__(self, mistfile=None, labels=["mini", "eep", "feh"],
                 predictions=["loga", "logl", "logt", "logg", "feh_surf"],
                 ageweight=True, verbose=True):

        # Initialize values.
        if mistfile is None:
            mistfile = minesweeper.__abspath__ + 'data/MIST/MIST_1.2_EEPtrk.h5'
        self.mistfile = mistfile

        self.labels = labels
        self.predictions = predictions
        self.ndim, self.npred = len(self.labels), len(self.predictions)

        self.null = np.zeros(self.npred) + np.nan

        # Import MIST grid.
        with h5py.File(self.mistfile, "r") as misth5:
            self.make_lib(misth5, verbose=verbose)
        self.lib_as_grid()

        # Construct age weights.
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
        self.output = [np.concatenate([misth5[z][p] for z in misth5["index"]])
                       for p in cols]
        self.output = np.array(self.output).T
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
        age_ind = self.predictions.index("loga")
        ageweights = np.zeros(len(self.libparams))
        for i, m in enumerate(self.gridpoints["mini"]):
            for j, z in enumerate(self.gridpoints["feh"]):
                if verbose:
                    sys.stderr.write("\rComputing age weights for track "
                                     "(mini, feh) = ({0}, {1})      "
                                     .format(m, z))
                    sys.stderr.flush()
                # Get indices for this track.
                inds = ((self.libparams["mini"] == m) &
                        (self.libparams["feh"] == z))
                # Store delta(ages). Assumes tracks are ordered by age.
                ageweights[inds] = np.gradient(10**self.output[inds, age_ind])

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

        self.grid_dims = np.append([len(self.gridpoints[p])
                                    for p in self.labels],
                                   self.output.shape[-1])
        self.xgrid = tuple([self.gridpoints[l] for l in self.labels])
        self.ygrid = np.zeros(self.grid_dims) + np.nan
        for x, y in zip(self.X, self.output):
            self.ygrid[tuple(x)] = y
        self.interpolator = RegularGridInterpolator(self.xgrid, self.ygrid)

    def get_predictions(self, labels):
        """
        Returns interpolated predictions for the input set of labels.

        Parameters
        ----------
        labels : 1-D or 2-D `~numpy.ndarray` of shape `(Nlabel, Nobj)`
            A set of labels we are interested in generating predictions for.

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
            preds = np.array([self.interpolator(l)[0] for l in labels])
        else:
            raise ValueError("Input `labels` not 1-D or 2-D.")

        return preds


class SEDmaker(MISTtracks):
    """
    An object that generates photometry interpolated from MIST tracks in
    EEP, initial mass, and metallicity using "The Payne".

    Parameters
    ----------
    filters : list of strings, optional
        The names of filters that photometry should be computed for. If not
        provided, photometry will be computed for all available filters.

    nnpath : str, optional
        The path to the neural network files from The Payne used to generate
        fast predictions. If not provided, these will be extracted from the
        `minesweeper` home path.

    mistfile : str, optional
        The name of the HDF5 file containing the MIST tracks. Default is
        `MIST_1.2_EEPtrk.h5` and is extracted from the `minesweeper`
        home path.

    labels : iterable of shape `(3)`, optional
        The names of the parameters over which to interpolate. This defaults to
        `["mini", "eep", "feh"]`.
        **Do not modify this unless you know what you're doing.**

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

    def __init__(self, filters=None, nnpath=None, mistfile=None,
                 labels=["mini", "eep", "feh"],
                 predictions=["loga", "logl", "logt", "logg", "feh_surf"],
                 ageweight=True, verbose=True):

        # Initialize filters.
        if filters is None:
            filters = FILTERS
        self.filters = filters
        if verbose:
            sys.stderr.write('Filters: {}\n'.format(filters))

        # Initialize underlying MIST tracks.
        super(SEDmaker, self).__init__(mistfile=mistfile, labels=labels,
                                       predictions=predictions,
                                       ageweight=ageweight, verbose=verbose)

        # Initialize The Payne.
        self.payne = FastPaynePredictor(filters=filters, nnpath=nnpath,
                                        verbose=verbose)

    def get_sed(self, mini=1., eep=350., feh=0., av=0.,
                dist=1000., loga_max=10.14, return_dict=True, **kwargs):
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

        av : float, optional
            Dust attenuation defined in terms of reddened V-band magnitudes.
            Default is `0.`.

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        return_dict : bool, optional
            Whether to return the parameters as a dictionary.
            Default is `True`.

        Returns
        -------
        sed : `~numpy.ndarray` of shape `(Nfilters,)`
            The predicted SED in magnitudes in the initialized filters.

        params : dict or array of length `(Npred,)`
            The corresponding predicted parameters associated with the given
            SED.

        """

        # Grab input labels.
        labels = {'mini': mini, 'eep': eep, 'feh': feh}  # establish dict
        labels = np.array([labels[l] for l in self.labels])  # reorder

        # Generate predictions.
        params_arr = self.get_predictions(labels)  # grab parameters
        params = dict(zip(self.predictions, params_arr))  # convert to dict

        if params['loga'] <= loga_max:
            # Compute SED.
            sed = self.payne.sed(logl=params["logl"], logt=params["logt"],
                                 logg=params["logg"],
                                 feh_surf=params["feh_surf"], alphafe=0.,
                                 av=av, dist=dist)
        else:
            # Generate "empty" SED.
            sed = np.full(self.payne.NFILT, np.nan)

        # If we are not returning a dictionary, overwrite `params`.
        if not return_dict:
            params = params_arr

        return sed, params

    def make_grid(self, mini_grid=None, eep_grid=None, feh_grid=None,
                  av_grid=None, av_wt=None, order=1,
                  dist=1000., loga_max=10.14, verbose=True, **kwargs):
        """
        Generate and return SED predictions over a grid in inputs.
        Reddened photometry is generated by fitting
        an nth-order polynomial in Av over the specified (weighted) Av grid,
        whose coefficients are stored.

        Parameters
        ----------
        mini_grid : `~numpy.ndarray`, optional
            Grid in initial mass (in units of solar masses). If not provided,
            the default is an adaptive grid with:
            (1) resolution of 0.02 from 0.3 to 2.8,
            (2) resolution of 0.1 from 2.8 to 3.0,
            (3) resolution of 0.25 from 3.0 to 8.0, and
            (4) resolution of 0.5 from 8.0 to 10.0.

        eep_grid : `~numpy.ndarray`, optional
            Grid in EEP. If not provided, the default is an adaptive grid with:
            (1) resolution of 12 from 202 to 454 (on the Main Sequence) and
            (2) resolution of 6 from 454 to 808 (off the Main Sequence).

        feh_grid : `~numpy.ndarray`, optional
            Grid in metallicity (defined logarithmically in units of solar
            metallicity). If not provided, the default is a grid from
            -4 to 0.5 with a resolution of 0.06.

        av_grid : `~numpy.ndarray`, optional
            Grid in dust attenuation defined in terms of reddened V-band
            magnitudes. Used to fit for a polynomial "reddening vector".
            If not provided, the default is a grid from 0. to 6.
            with a resolution of 0.3.

        av_wt : `~numpy.ndarray`, optional
            The associated weights over the provided `av_grid` to be used when
            fitting. If not provided, the default is `(1e-5 + av_grid)**-1`.
            This forces the fit to go through `Av=0.` with 1/x weighting
            for the remaining points.

        order : int, optional
            Order of the polynomial approximation used for the reddening
            vector. Default is `1` (linear).

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        verbose : bool, optional
            Whether to print progress. Default is `True`.

        """

        # Initialize grid.
        labels = ['mini', 'eep', 'feh']
        ltype = np.dtype([(n, np.float) for n in labels])
        if mini_grid is None:  # initial mass
            mini_grid = np.concatenate([np.arange(0.3, 2.8, 0.02),
                                        np.arange(2.8, 3. + 1e-5, 0.1),
                                        np.arange(3.25, 8., 0.25),
                                        np.arange(8., 10. + 1e-5, 0.5)])
        if eep_grid is None:  # EEP
            eep_grid = np.concatenate([np.arange(202, 454, 12),
                                       np.arange(454, 808, 6)])
        if feh_grid is None:  # metallicity
            feh_grid = np.arange(-4., 0.5 + 1e-5, 0.06)
            feh_grid[-1] -= 1e-5
        if av_grid is None:  # reddening
            av_grid = np.arange(0., 6. + 1e-5, 0.3)
            av_grid[-1] -= 1e-5
        if av_wt is None:  # Av weights
            # Pivot around Av=0 point with inverse Av weighting.
            av_wt = (1e-5 + av_grid)**-1.

        self.grid_label = np.array(list(product(*[mini_grid, eep_grid,
                                                  feh_grid])),
                                   dtype=ltype)
        Ngrid = len(self.grid_label)

        # Generate SEDs on the grid.
        ptype = np.dtype([(n, np.float) for n in self.predictions])
        stype = np.dtype([(n, np.float, order + 1) for n in self.filters])
        self.grid_sed = np.full(Ngrid, np.nan, dtype=stype)
        self.grid_param = np.full(Ngrid, np.nan, dtype=ptype)
        self.grid_sel = np.ones(Ngrid, dtype='bool')

        percentage = -99
        for i, (mini, eep, feh) in enumerate(self.grid_label):
            # Print progress.
            new_percentage = int((i+1) / Ngrid * 1e4)
            if verbose and new_percentage != percentage:
                percentage = new_percentage
                sys.stderr.write('\rConstructing grid {:6.2f}% ({:d}/{:d}) '
                                 '[mini={:6.3f}, eep={:6.3f}, feh={:6.3f}] '
                                 '           '
                                 .format(percentage / 100., i+1, Ngrid,
                                         mini, eep, feh))
                sys.stderr.flush()

            # Compute model and parameter predictions.
            sed, params = self.get_sed(mini=mini, eep=eep, feh=feh,
                                       av=0., dist=dist, loga_max=loga_max,
                                       return_dict=False)
            self.grid_param[i] = tuple(params)

            # Deal with non-existent SEDS.
            if np.any(np.isnan(sed)) or np.any(np.isnan(params)):
                # Flag results and fill with `nan`s.
                self.grid_sel[i] = False
                self.grid_sed[i] = tuple(np.full((self.payne.NFILT, order),
                                                 np.nan))
            else:
                # Compute polynomial fit.
                # TODO: MODIFY TO MATCH [m, R0, dRdRv] framework for Bayestar
                seds = np.array([self.get_sed(mini=mini, eep=eep, feh=feh,
                                              av=av, dist=dist,
                                              loga_max=loga_max,
                                              return_dict=False)[0]
                                 for av in av_grid])
                self.grid_sed[i] = tuple(polyfit(av_grid, seds,
                                                 order, w=av_wt).T)

        if verbose:
            sys.stderr.write('\n')


class FastNN(object):
    """
    Object that wraps the underlying neural networks used to train "The Payne".

    Parameters
    ----------
    nnlist : list of strings
        List of filenames where the neural networks are stored.

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    """

    def __init__(self, nnlist, verbose=True):

        # Initialize values.
        if verbose:
            sys.stderr.write('Initializing FastNN predictor...')
        self._convert_torch(nnlist)
        self.set_minmax(nnlist)
        if verbose:
            sys.stderr.write('done!\n')

    def _convert_torch(self, nnlist):
        """
        Convert `torch.Variable` to `~numpy.ndarray` of approriate shape.

        Parameters
        ----------
        nnlist : list of strings
            List of filenames where the neural networks are stored.

        """

        # Store weights and bias.
        self.w1 = np.array([nn.model.lin1.weight.data.numpy()
                            for nn in nnlist])
        self.b1 = np.expand_dims(np.array([nn.model.lin1.bias.data.numpy()
                                           for nn in nnlist]), -1)
        self.w2 = np.array([nn.model.lin2.weight.data.numpy()
                            for nn in nnlist])
        self.b2 = np.expand_dims(np.array([nn.model.lin2.bias.data.numpy()
                                           for nn in nnlist]), -1)
        self.w3 = np.array([nn.model.lin3.weight.data.numpy()
                            for nn in nnlist])
        self.b3 = np.expand_dims(np.array([nn.model.lin3.bias.data.numpy()
                                           for nn in nnlist]), -1)

    def set_minmax(self, nnlist):
        """
        Set the values necessary for scaling/encoding the feature vector and
        make sure they are the same for every pixel/band.

        Parameters
        ----------
        nnlist : list of strings
            List of filenames where the neural networks are stored.

        """

        # Check if `nnlist` is non-empty.
        try:
            nn = nnlist[0]
            self.xmin = nn.model.xmin
            self.xmax = nn.model.xmax
        except:
            raise ValueError("Could not find an appropriate `xmin, xmax` for "
                             "scaling `x` variable")

        # Check that all NNs have the same `xspan`.
        self.xspan = (self.xmax - self.xmin)
        assert np.all(self.xspan > 0)
        for nn in nnlist:
            assert np.all(nn.model.xmin == self.xmin)
            assert np.all(nn.model.xmax == self.xmax)

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


class FastPaynePredictor(FastNN):
    """
    Object that generates SED predictions for a provided set of filters
    using the `minesweeper` neural networks used to train "The Payne".

    Parameters
    ----------
    filters : list of strings
        The names of filters that photometry should be computed for.

    nnpath : str, optional
        The path to the neural network directory.

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    """

    def __init__(self, filters, nnpath=None, verbose=True):

        # Initialize values.
        self.filters = filters
        self.NFILT = len(filters)
        nnlist = [ANN(f, nnpath=nnpath, verbose=False) for f in filters]
        super(FastPaynePredictor, self).__init__(nnlist, verbose=verbose)

    def sed(self, logt=3.8, logg=4.4, feh_surf=0., logl=0., alphafe=0., av=0.,
            dist=1000., filt_idxs=slice(None)):
        """
        Returns the SED predicted by "The Payne" for the input set of
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

        alphafe : float, optional
            The alpha enhancement in logarithmic units relative to solar
            values. Default is `0.`.

        av : float, optional
            Dust attenuation in units of V-band reddened magnitudes.
            Default is `0.`.

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
        x = np.array([10.**logt, logg, feh_surf, alphafe, av])
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
