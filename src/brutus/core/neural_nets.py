#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network utilities for fast SED prediction.

This module contains classes for neural network-based bolometric correction
predictions, enabling fast computation of stellar spectral energy distributions
from fundamental stellar parameters (Teff, log g, [Fe/H], [α/Fe], Av, Rv).

The neural networks are trained on synthetic stellar spectra and provide rapid
computation of bolometric corrections across multiple photometric bands,
significantly speeding up stellar parameter inference compared to full spectral
synthesis.

Classes
-------
FastNN : Base neural network class
    Provides core neural network functionality for bolometric correction
    prediction using pre-trained weights and biases.

FastNNPredictor : SED prediction class
    Extends FastNN to generate complete spectral energy distributions
    for multiple filters with distance modulus and reddening corrections.

Examples
--------
Basic usage for SED prediction:

>>> from brutus.core.neural_nets import FastNNPredictor
>>> from brutus.data.filters import FILTERS
>>>
>>> # Initialize predictor for specific filters
>>> predictor = FastNNPredictor(filters=['g', 'r', 'i', 'z'])
>>>
>>> # Predict SED for a solar-type star at 1 kpc
>>> sed = predictor.sed(logt=3.76, logg=4.44, feh_surf=0.0,
...                     logl=0.0, afe=0.0, av=0.1, rv=3.1, dist=1000)
>>> print(f"Predicted magnitudes: {sed}")

Notes
-----
The neural networks require pre-trained model files containing weights, biases,
and input scaling parameters. Default files are stored in the data directory
and are automatically downloaded when needed.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

# Import filter definitions from parent module
from ..data.filters import FILTERS

__all__ = ["FastNN", "FastNNPredictor"]


class FastNN:
    """
    Object that wraps the underlying neural networks used to interpolate.

    between grid points on the bolometric correction tables.

    This class provides the core neural network functionality for predicting
    bolometric corrections from stellar parameters. It loads pre-trained
    neural network weights and biases, and provides methods for encoding
    input parameters and evaluating the network.

    Parameters
    ----------
    filters : list of str, optional
        The names of filters that photometry should be computed for.
        If not provided, all available filters will be used. Filter names
        should match those defined in `brutus.data.filters.FILTERS`.

    nnfile : str, optional
        Path to the neural network file containing pre-trained weights
        and biases. Default is `'brutus/data/DATAFILES/nnMIST_BC.h5'` which will
        be downloaded automatically if not present.

    verbose : bool, optional
        Whether to print initialization progress messages to stderr.
        Default is `True`.

    Attributes
    ----------
    w1, w2, w3 : numpy.ndarray
        Neural network weight matrices for each layer.

    b1, b2, b3 : numpy.ndarray
        Neural network bias vectors for each layer.

    xmin, xmax : numpy.ndarray
        Minimum and maximum values for input parameter scaling.

    xspan : numpy.ndarray
        Range of input parameters (xmax - xmin).

    Notes
    -----
    The neural network architecture is a 3-layer feedforward network with
    sigmoid activation functions. Input parameters are scaled to [0,1] range
    before evaluation.

    Expected input parameters (in order):
    - log10(Teff) : Effective temperature in Kelvin
    - log g : Surface gravity in cgs units
    - [Fe/H] : Surface metallicity (log scale)
    - [α/Fe] : Alpha enhancement (log scale)
    - Av : V-band extinction in magnitudes
    - Rv : Reddening parameter R(V) = A(V)/E(B-V)
    """

    def __init__(self, filters=None, nnfile=None, verbose=True):

        # Initialize values.
        if filters is None:
            filters = np.array(FILTERS)
        # TODO: Extract shared file discovery logic to a utility function
        if nnfile is None:
            import os

            package_root = Path(
                __file__
            ).parent.parent.parent.parent  # Get the package root directory

            # Try multiple possible names (nn_c3k.h5 is downloaded by pooch,
            # nnMIST_BC.h5 is legacy name - they are the same file)
            possible_names = ["nn_c3k.h5", "nnMIST_BC.h5"]

            for nn_name in possible_names:
                # Check local data directory first
                nnfile = package_root / "data" / "DATAFILES" / nn_name
                if os.path.exists(str(nnfile)):
                    break

                # If not found locally, try pooch cache directory
                import pooch

                cache_dir = Path(pooch.os_cache("astro-brutus"))
                cache_path = cache_dir / nn_name
                if os.path.exists(str(cache_path)):
                    nnfile = cache_path
                    break

        # Read in NN data.
        if verbose:
            sys.stderr.write("Initializing FastNN predictor...")
        self._load_NN(filters, nnfile)
        if verbose:
            sys.stderr.write("done!\n")

    def _load_NN(self, filters, nnfile):
        """
        Load neural network weights and biases from HDF5 file.

        This method reads the pre-trained neural network parameters for each
        specified filter from the HDF5 file. Each filter has its own trained
        network with identical architecture but different weights.

        Parameters
        ----------
        filters : array-like
            List of filter names to load networks for.

        nnfile : str, optional
            Path to HDF5 file containing neural network data.

        Raises
        ------
        ValueError
            If neural networks have inconsistent input parameter ranges
            across different filters.

        Notes
        -----
        The HDF5 file is expected to have the following structure:
        - /{filter}/w1, w2, w3 : weight matrices for layers 1, 2, 3
        - /{filter}/b1, b2, b3 : bias vectors for layers 1, 2, 3
        - /{filter}/xmin, xmax : input parameter scaling bounds
        """
        with h5py.File(nnfile, "r") as f:
            # Store weights and bias for each layer and filter
            self.w1 = np.array([f[fltr]["w1"] for fltr in filters])
            self.b1 = np.array([f[fltr]["b1"] for fltr in filters])
            self.w2 = np.array([f[fltr]["w2"] for fltr in filters])
            self.b2 = np.array([f[fltr]["b2"] for fltr in filters])
            self.w3 = np.array([f[fltr]["w3"] for fltr in filters])
            self.b3 = np.array([f[fltr]["b3"] for fltr in filters])

            # Load input parameter scaling bounds
            xmin = np.array([f[fltr]["xmin"] for fltr in filters])
            xmax = np.array([f[fltr]["xmax"] for fltr in filters])

            # Verify all networks have consistent parameter ranges
            if len(np.unique(xmin)) == 6 and len(np.unique(xmax)) == 6:
                self.xmin = xmin[0]
                self.xmax = xmax[0]
                self.xspan = self.xmax - self.xmin
            else:
                raise ValueError(
                    "Some of the neural networks have different "
                    "`xmin` and `xmax` ranges for parameters."
                )

    def encode(self, x):
        """
        Rescale input parameters to [0,1] range for neural network evaluation.

        The neural networks are trained on scaled inputs where each parameter
        is normalized to the range [0,1] based on the training data bounds.

        Parameters
        ----------
        x : numpy.ndarray of shape (Ninput,) or (Ninput, Nsamples)
            Input stellar parameters. Expected parameters are:
            [log10(Teff), log g, [Fe/H], [alpha/Fe], Av, Rv]

        Returns
        -------
        xp : numpy.ndarray of shape (Ninput, 1) or (Ninput, Nsamples)
            Scaled input parameters ready for neural network evaluation.

        Notes
        -----
        The scaling is applied as: x_scaled = (x - xmin) / (xmax - xmin)
        where xmin and xmax are the bounds from the training data.
        """
        try:
            # Handle 1D input case
            xp = (np.atleast_2d(x) - self.xmin[None, :]) / self.xspan[None, :]
            return xp.T
        except (ValueError, IndexError):
            # Handle 2D input case (multiple evaluations)
            xp = (np.atleast_2d(x) - self.xmin[:, None]) / self.xspan[:, None]
            return xp

    def sigmoid(self, a):
        """
        Apply sigmoid activation function.

        Computes the logistic sigmoid function: f(a) = 1 / (1 + exp(-a))

        Parameters
        ----------
        a : numpy.ndarray
            Input array to apply sigmoid transformation to.

        Returns
        -------
        a_t : numpy.ndarray
            Output after applying sigmoid activation, same shape as input.

        Notes
        -----
        The sigmoid function maps any real number to the range (0, 1),
        providing smooth activation for the neural network hidden layers.
        """
        return 1.0 / (1.0 + np.exp(-a))

    def nneval(self, x):
        """
        Evaluate the neural network for given input parameters.

        Performs forward propagation through the 3-layer neural network
        to predict bolometric corrections for all filters.

        Parameters
        ----------
        x : numpy.ndarray of shape (Ninput,)
            Stellar parameters: [log10(Teff), log g, [Fe/H], [alpha/Fe], Av, Rv]

        Returns
        -------
        y : numpy.ndarray
            Predicted bolometric corrections for each filter.

        Notes
        -----
        The network architecture is:
        - Input layer: 6 parameters
        - Hidden layer 1: with sigmoid activation
        - Hidden layer 2: with sigmoid activation
        - Output layer: linear activation (bolometric corrections)
        """
        # Forward propagation through the network
        a1 = self.sigmoid(np.matmul(self.w1, self.encode(x)) + self.b1)
        a2 = self.sigmoid(np.matmul(self.w2, a1) + self.b2)
        y = np.matmul(self.w3, a2) + self.b3

        return np.squeeze(y)


class FastNNPredictor(FastNN):
    """
    Object that generates SED predictions for a provided set of filters using neural networks.

    This class extends FastNN to provide a complete interface for stellar
    SED prediction, including automatic distance modulus calculation and
    conversion from bolometric corrections to apparent magnitudes.

    Parameters
    ----------
    filters : list of str, optional
        The names of filters that photometry should be computed for.
        If not provided, all available filters will be used. Must be
        a subset of filters available in the neural network file.

    nnfile : str, optional
        Path to the neural network file containing pre-trained weights.
        Default is `'brutus/data/DATAFILES/nnMIST_BC.h5'` which contains networks
        trained on MIST isochrones with C3K synthetic spectra.

    verbose : bool, optional
        Whether to print initialization progress messages. Default is `True`.

    Attributes
    ----------
    filters : numpy.ndarray
        Array of filter names for which predictions are made.

    NFILT : int
        Number of filters for which predictions are made.

    Examples
    --------
    Predict SED for a solar analog:

    >>> predictor = FastNNPredictor(filters=['g', 'r', 'i'])
    >>> sed = predictor.sed(logt=3.76, logg=4.44, feh_surf=0.0,
    ...                     logl=0.0, dist=1000.)
    >>> print(f"g-r color: {sed[0] - sed[1]:.3f}")

    Predict for a red giant with extinction:

    >>> sed = predictor.sed(logt=3.60, logg=2.5, feh_surf=-0.5,
    ...                     logl=1.5, av=0.5, rv=3.1, dist=2000.)

    Notes
    -----
    The neural networks provide bolometric corrections which are combined
    with luminosity and distance to produce apparent magnitudes:

    m = -2.5 * log10(L/L_sun) + 4.74 - BC + distance_modulus

    where BC is the bolometric correction predicted by the neural network.
    """

    def __init__(self, filters=None, nnfile=None, verbose=True):

        # Initialize filter selection
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters
        self.NFILT = len(filters)

        # Set default neural network file
        # TODO: Extract shared file discovery logic to a utility function
        if nnfile is None:
            import os

            package_root = Path(
                __file__
            ).parent.parent.parent.parent  # Get the package root directory

            # Try multiple possible names (nn_c3k.h5 is downloaded by pooch,
            # nnMIST_BC.h5 is legacy name - they are the same file)
            possible_names = ["nn_c3k.h5", "nnMIST_BC.h5"]

            for nn_name in possible_names:
                # Check local data directory first
                nnfile = package_root / "data" / "DATAFILES" / nn_name
                if os.path.exists(str(nnfile)):
                    break

                # If not found locally, try pooch cache directory
                import pooch

                cache_dir = Path(pooch.os_cache("astro-brutus"))
                cache_path = cache_dir / nn_name
                if os.path.exists(str(cache_path)):
                    nnfile = cache_path
                    break

        # Initialize parent class with neural network
        super().__init__(filters=filters, nnfile=nnfile, verbose=verbose)

    def sed(
        self,
        logt=3.8,
        logg=4.4,
        feh_surf=0.0,
        logl=0.0,
        afe=0.0,
        av=0.0,
        rv=3.3,
        dist=1000.0,
        filt_idxs=slice(None),
    ):
        """
        Generate SED predictions for specified stellar parameters.

        Returns predicted apparent magnitudes in the specified filters
        for a star with the given physical parameters, distance, and
        extinction. Uses neural network bolometric corrections combined
        with standard photometric transformations.

        Parameters
        ----------
        logt : float, optional
            Base-10 logarithm of effective temperature in Kelvin.
            Typical range: [3.3, 4.5] corresponding to ~2000-30000K.
            Default is 3.8 (6300K, solar-type).

        logg : float, optional
            Base-10 logarithm of surface gravity in cgs units (cm/s^2).
            Typical range: [0, 5] from supergiants to white dwarfs.
            Default is 4.4 (solar value).

        feh_surf : float, optional
            Surface metallicity [Fe/H] in logarithmic units relative to solar.
            Typical range: [-2.5, 0.5]. Default is 0.0 (solar).

        logl : float, optional
            Base-10 logarithm of luminosity in solar luminosities.
            Typical range: [-4, 6] from low-mass MS to supergiants.
            Default is 0.0 (solar luminosity).

        afe : float, optional
            Alpha element enhancement [alpha/Fe] in logarithmic units relative
            to solar abundance ratios. Typical range: [-0.2, 0.8].
            Default is 0.0 (solar ratios).

        av : float, optional
            V-band extinction in magnitudes. Must be non-negative.
            Typical range: [0, 6] mag. Default is 0.0 (no extinction).

        rv : float, optional
            Reddening parameter R(V) = A(V)/E(B-V), describing the
            extinction curve shape. Typical range: [1, 8].
            Default is 3.3 (Milky Way average).

        dist : float, optional
            Distance to the star in parsecs. Must be positive.
            Default is 1000 pc.

        filt_idxs : slice or array-like, optional
            Indices or slice object specifying which subset of filters
            to return predictions for. Default is slice(None) (all filters).

        Returns
        -------
        sed : numpy.ndarray of shape (Nfilt_subset,)
            Predicted apparent magnitudes in the specified filter subset.
            Magnitudes are in the AB system and include distance modulus
            and extinction corrections.

        Notes
        -----
        The computation follows these steps:

        1. Compute distance modulus: mu = 5 * log10(dist) - 5
        2. Evaluate neural network for bolometric corrections: BC = NN(params)
        3. Convert to apparent magnitudes: m = -2.5 * logl + 4.74 - BC + mu

        If any input parameters are outside the neural network training
        bounds, NaN values are returned for safety.

        Examples
        --------
        Solar analog at various distances:

        >>> predictor = FastNNPredictor(['V', 'K'])
        >>> for d in [100, 1000, 10000]:  # pc
        ...     sed = predictor.sed(dist=d)
        ...     print(f"{d:5d} pc: V={sed[0]:.2f}, K={sed[1]:.2f}")

        Effect of extinction:

        >>> sed_clean = predictor.sed(av=0.0)
        >>> sed_dusty = predictor.sed(av=1.0, rv=3.1)
        >>> extinction = sed_dusty - sed_clean
        >>> print(f"V-band extinction: {extinction[0]:.2f} mag")
        """
        # Compute distance modulus
        mu = 5.0 * np.log10(dist) - 5.0

        # Prepare input parameters for neural network
        x = np.array([10.0**logt, logg, feh_surf, afe, av, rv])

        # Check if parameters are within neural network bounds
        if np.all(np.isfinite(x)) and np.all((x >= self.xmin) & (x <= self.xmax)):
            # Parameters are valid - compute bolometric corrections
            BC = self.nneval(x)

            # Convert to apparent magnitudes
            # m = M_bol + BC + distance_modulus
            # where M_bol = -2.5*log10(L/L_sun) + M_bol_sun
            # and M_bol_sun = 4.74
            m = -2.5 * logl + 4.74 - BC + mu
        else:
            # Parameters are out of bounds - return NaN values
            m = np.full(self.NFILT, np.nan)

        # Return specified subset of filters
        return np.atleast_1d(m)[filt_idxs]

    def sed_batch(
        self,
        logt,
        logg,
        feh_surf,
        logl,
        afe,
        av,
        rv,
        dist,
        filt_idxs=slice(None),
    ):
        """
        Generate SED predictions for an array of stellar parameters.

        Vectorized version of :meth:`sed` that evaluates the neural network
        once for all stars, providing significant speedup over looping.

        Parameters
        ----------
        logt : numpy.ndarray of shape (N,)
            Base-10 logarithm of effective temperature for each star.
        logg : numpy.ndarray of shape (N,)
            Base-10 logarithm of surface gravity for each star.
        feh_surf : numpy.ndarray of shape (N,)
            Surface metallicity [Fe/H] for each star.
        logl : numpy.ndarray of shape (N,)
            Base-10 logarithm of luminosity for each star.
        afe : numpy.ndarray of shape (N,)
            Alpha element enhancement [alpha/Fe] for each star.
        av : float
            V-band extinction (same for all stars).
        rv : float
            Reddening parameter R(V) (same for all stars).
        dist : float
            Distance in parsecs (same for all stars).
        filt_idxs : slice or array-like, optional
            Filter subset to return. Default is all filters.

        Returns
        -------
        seds : numpy.ndarray of shape (N, Nfilt_subset)
            Predicted apparent magnitudes for each star and filter.
            Stars with out-of-bounds parameters get NaN values.
        """
        logt = np.asarray(logt)
        logg = np.asarray(logg)
        feh_surf = np.asarray(feh_surf)
        logl = np.asarray(logl)
        afe = np.asarray(afe)
        N = len(logt)

        # Distance modulus (constant for all stars)
        mu = 5.0 * np.log10(dist) - 5.0

        # Build input array: shape (6, N)
        x = np.array([10.0**logt, logg, feh_surf, afe, np.full(N, av), np.full(N, rv)])

        # Determine which stars have valid (in-bounds, finite) parameters
        valid = (
            np.all(np.isfinite(x), axis=0)
            & np.all(x >= self.xmin[:, None], axis=0)
            & np.all(x <= self.xmax[:, None], axis=0)
        )  # shape (N,)

        # Initialize output with NaN
        seds = np.full((N, self.NFILT), np.nan)

        n_valid = np.sum(valid)
        if n_valid > 0:
            # Evaluate NN for all valid stars at once
            BC = self.nneval(x[:, valid])  # shape (NFILT, n_valid)

            # Convert to apparent magnitudes
            seds[valid] = -2.5 * logl[valid, None] + 4.74 - BC.T + mu

        return seds[:, filt_idxs]
