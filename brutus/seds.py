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

import minesweeper
from minesweeper.photANN import ANN

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

# Define the set of filters was have available.
gaia = ["Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev"]
sdss = ["SDSS_{}".format(b) for b in "ugriz"]
ps = ["PS_{}".format(b) for b in "grizy"]
decam = ["DECam_{}".format(b) for b in "ugrizY"]
tycho = ["Tycho_B", "Tycho_V"]
bessell = ["Bessell_{}".format(b) for b in "UBVRI"]
tmass = ["2MASS_{}".format(b) for b in ["J", "H", "Ks"]]
wise = ["WISE_W{}".format(b) for b in "1234"]
galex = ["GALEX_NUV", "GALEX_FUV"]

FILTERS = gaia + sdss + ps + tmass + wise + galex

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

    labels : iterable of shape (3), optional
        The names of the parameters on which to interpolate. This defaults to
        `["mini", "eep", "feh"]`. **Change this only if you know what**
        **you're doing.**

    predictions : iterable of shape (4), optional
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
        are added to the output set of predictions.

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
                                     "(mini, feh)=({0}, {1})      "
                                     .format(m, z))
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
        used to generate fast predictions.

        """

        self.grid_dims = np.append([len(self.gridpoints[p])
                                    for p in self.labels],
                                   self.output.shape[-1])
        self.xgrid = tuple([self.gridpoints[l] for l in self.gridpoints])
        self.ygrid = np.zeros(self.grid_dims)
        for x, y in zip(self.X, self.output):
            self.ygrid[tuple(x)] = y
        self.interpolator = RegularGridInterpolator(self.xgrid, self.ygrid)

    def get_predictions(self, labels):
        """
        Returns interpolated predictions for the input set of labels.

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
    EEP, initial mass, and metallicity using The Payne.

    Parameters
    ----------
    filters : list of strings
        The names of filters that photometry should be computed for.

    nnpath : str, optional
        The path to the neural network files from The Payne used to generate
        fast predictions. If not provided, these will be extracted from the
        `minesweeper` home path.

    mistfile : str, optional
        The name of the HDF5 file containing the MIST tracks. Default is
        `MIST_1.2_EEPtrk.h5` and is extracted from the `minesweeper`
        home path.

    labels : iterable of shape (3), optional
        The names of the parameters on which to interpolate. This defaults to
        `["mini", "eep", "feh"]`. **Change this only if you know what**
        **you're doing.**

    predictions : iterable of shape (4), optional
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

    def __init__(self, filters, nnpath=None, mistfile=None,
                 labels=["mini", "eep", "feh"],
                 predictions=["loga", "logl", "logt", "logg", "feh_surf"],
                 ageweight=True, verbose=True):

        # Initialize underlying MIST tracks.
        super(SEDmaker, self).__init__(mistfile=mistfile, labels=labels,
                                       predictions=predictions,
                                       ageweight=ageweight, verbose=verbose)

        self.filters = filters
        self.payne = FastPaynePredictor(filters=filters, nnpath=nnpath,
                                        verbose=verbose)

    def get_sed(self, mini=1.0, eep=350, feh=0., av=0., dist=1000.,
                return_dict=True, **kwargs):
        """
        Generate and return SED predictions for input initial mass (`mini`),
        EEP (`eep`), metallicity (`feh`), reddening (`av`), and distance in
        pc (`dist`). Returns the SED and associated parameters.

        """

        # Grab input labels.
        labels = {'mini': mini, 'eep': eep, 'feh': feh}  # establish dict
        labels = np.array([labels[l] for l in self.labels])  # reorder

        # Generate predictions.
        params_arr = self.get_predictions(labels)  # grab parameters
        params = dict(zip(self.predictions, params_arr))  # convert to dict

        # Compute SED.
        sed = self.payne.sed(logl=params["logl"], logt=params["logt"],
                             logg=params["logg"], feh_surf=params["feh_surf"],
                             av=av, dist=dist)

        if return_dict:
            return sed, params
        else:
            return sed, params_arr

    def get_grid(self, mini_grid=None, eep_grid=None, feh_grid=None,
                 av_grid=None, dist=1000., verbose=True, **kwargs):
        """
        Generate and return SED predictions over a grid in initial mass,
        EEP, metallicity, and reddening. Note that non-physical models are
        automatically removed. Returns `sed_grid`, `input_grid`, and
        `output_grid`.

        """

        # Initialize grid.
        labels = ['mini', 'eep', 'feh', 'av']
        dtype = np.dtype([(n, np.float) for n in labels])
        if mini_grid is None:
            mini_grid = np.arange(0.5, 10. + 1e-5, 0.05)
        if eep_grid is None:
            eep_grid = np.arange(202, 808 + 1e-5, 2)
        if feh_grid is None:
            feh_grid = np.arange(-2., 0.5 + 1e-5, 0.25)
        if av_grid is None:
            av_grid = np.concatenate([np.arange(0., 1., 0.025),
                                      np.arange(1., 2. + 1e-5, 0.05),
                                      np.arange(2., 3. + 1e-5, 0.1)])

        grid = np.array(list(product(*[mini_grid, eep_grid, feh_grid,
                                       av_grid])), dtype=dtype)
        Ngrid = len(grid)

        # Generate SEDs on the grid.
        rtype = np.dtype([(n, np.float) for n in self.predictions])
        sed_grid = np.zeros((Ngrid, len(self.filters)))
        param_grid = np.zeros(Ngrid, dtype=rtype)
        sel = np.ones_like(Ngrid, dtype='bool')

        percentage = -99
        for i, (mini, eep, feh, av) in enumerate(grid):
            # Print progress.
            new_percentage = int((i+1) / Ngrid * 1e5)
            if verbose and new_percentage != percentage:
                percentage = new_percentage
                sys.stderr.write('\rConstructing grid {0}% '
                                 .format(percentage / 100.))
                sys.stderr.flush()

            # Compute model and parameter predictions.
            sed, params = self.get_sed(mini=mini, eep=eep, feh=feh, av=av,
                                       dist=dist, return_dict=False)
            sed_grid[i], param_grid[i] = sed, params

            # Flag bad results.
            if np.any(np.isnan(sed)) or np.any(np.isnan(params)):
                sel[i] = False

        if verbose:
            sys.stderr.write('\n')

        return sed_grid[sel], grid[sel], param_grid[sel]


class FastNN(object):
    """
    Object that wraps the underlying neural networks used to train The Payne.

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
        Rescale the `x` iterable. Returns an `~numpy.ndarray` of
        shape (Nfilt, 1).

        """

        try:
            xp = (np.atleast_2d(x) - self.xmin[None, :]) / self.xspan[None, :]
            return xp.T
        except:
            xp = (np.atleast_2d(x) - self.xmin[:, None]) / self.xspan[:, None]
            return xp

    def sigmoid(self, a):
        """
        Evaluate the sigmoid of `a`.

        """

        return 1. / (1 + np.exp(-a))

    def nneval(self, x):
        """
        Evaluate the neural network at the value of `x`.

        """

        a1 = self.sigmoid(np.matmul(self.w1, self.encode(x)) + self.b1)
        a2 = self.sigmoid(np.matmul(self.w2, a1) + self.b2)
        y = np.matmul(self.w3, a2) + self.b3

        return np.squeeze(y)


class FastPaynePredictor(FastNN):
    """
    Object that generates SED predictions for a provided set of filters
    using the `minesweeper` neural networks used to train The Payne.

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
        nnlist = [ANN(f, nnpath=nnpath, verbose=False) for f in filters]
        super(FastPaynePredictor, self).__init__(nnlist, verbose=verbose)

    def sed(self, logt=3.8, logg=4.4, feh_surf=0., logl=0.0, av=0.0,
            dist=1000.0, filt_idxs=slice(None)):
        """
        Returns the SED predicted by The Payne for the input set of
        physical parameters for a specified subset of bands. Predictions
        are in apparent magnitudes at the specified distance. See
        `self.filters` for the order of the bands.

        """

        # Compute distance modulus.
        mu = 5 * np.log10(dist) - 5

        # Compute bolometric correction.
        BC = self.nneval([10.**logt, logg, feh_surf, av])

        # Compute apparent magnitudes.
        m = -2.5 * logl + 4.74 - BC + mu

        return np.atleast_1d(m)[filt_idxs]
