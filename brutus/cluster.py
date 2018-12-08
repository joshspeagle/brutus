#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Star cluster fitting utilities.

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
from scipy.interpolate import LinearNDInterpolator as LNDInterp

from .utils import get_seds, add_mag

__all__ = ["Isochrone"]


class Isochrone(object):
    """
    Object that returns photometry for a predicted isochrone.

    Parameters
    ----------
    models : variable input
        The input collection of models that will be used to generate
        predicted isochrones. This can either be a 2-tuple that contains
        `(mags, labels)` following the same format passed to
        `~brutus.fitting.BruteForce` or a `~brutus.seds.SEDmaker` object.

    mini_lims : 2-tuple, optional
        Limits on the range of allowed initial mass. Default is `(0.1, 10.)`.

    feh_lims : 2-tuple, optional
        Limits on the range of allowed metallicities. Default is `(-4., 0.5)`.

    eep_lims : 2-tuple, optional
        Limits on the range of allowed EEPs. Default is `(202, 808)`.

    loga_lims : 2-tuple, optional
        Limits on the range of allowed log(ages). Default is `(5., 10.14)`.

    """

    def __init__(self, models, mini_lims=(0.1, 10.), feh_lims=(-4., 0.5),
                 eep_lims=(202, 808), loga_lims=(5., 10.14)):

        self.lims_mini = mini_lims
        self.lims_feh = feh_lims
        self.lims_eep = eep_lims
        self.lims_loga = loga_lims

        try:
            # Check if we have been provided a pre-generated grid
            # of photometry and labels.
            self.mags, self.labels = models

            # Grab corresponding grids in initial mass (`mini`), metallicity
            # (`feh`), and "equivalent evolutionary points" (`eep`).
            mini, feh, eep, loga = (self.labels['mini'], self.labels['feh'],
                                    self.labels['eep'], self.labels['loga'])
            sel = ((mini >= mini_lims[0]) & (mini <= mini_lims[1]) &
                   (feh >= feh_lims[0]) & (feh <= feh_lims[1]) &
                   (eep >= eep_lims[0]) & (eep <= eep_lims[1]) &
                   (loga >= loga_lims[0]) & (loga <= loga_lims[1]))
            mini, feh, loga = mini[sel], feh[sel], loga[sel]
            self.mags, self.labels = self.mags[sel], self.labels[sel]
            self.mgrid = np.unique(mini)

            # Grab corresponding labels used to generate photometry.
            logt, logg, logl = (self.labels['logt'], self.labels['logg'],
                                self.labels['logl'])
            self.pred = np.c_[logt, logg, feh, logl, eep]

            # Initialize interpolator.
            self.interpolator = LNDInterp(np.c_[mini, feh, loga], self.mags)
            self.interpolator_p = LNDInterp(np.c_[mini, feh, loga], self.pred)
            self.itype = 'grid'
        except:
            # Save SEDMaker object.
            self.sedmaker = models

            # Grab corresponding grids in initial mass (`mini`), metallicity
            # (`feh`), and "equivalent evolutionary points" (`eep`).
            age_idx = np.where('loga' == np.array(models.predictions))[0][0]
            mini, feh, eep, loga = (models.libparams['mini'],
                                    models.libparams['feh'],
                                    models.libparams['eep'],
                                    models.output[:, age_idx])
            sel = ((mini >= mini_lims[0]) & (mini <= mini_lims[1]) &
                   (feh >= feh_lims[0]) & (feh <= feh_lims[1]) &
                   (eep >= eep_lims[0]) & (eep <= eep_lims[1]))
            mini, feh, loga = mini[sel], feh[sel], loga[sel]
            self.mgrid = np.unique(mini)
            self.itype = 'sedmaker'

    def _interpolate_grid(self, loga=9., feh=0., av=0., rv=3.3, smf=0.,
                          dist=1000., loga_max=10.14, filt_idxs=slice(None),
                          mgrid=None, return_params=False):
        """
        Returns a predicted isochrone for the input set of
        parameters for a specified subset of bands. Predictions
        are in apparent magnitudes at the specified distance. Based on
        interpolating over a **pre-generated grid** of input photometry
        using `~scipy.interpolate.LinearNDInterpolator`.

        Parameters
        ----------
        loga : float
            The base-10 logarithm of the isochrone age in years.
            Default is `9.`.

        feh : float, optional
            The metallicity defined logarithmically relative to the solar value.
            Default is `0.`

        av : float, optional
            Dust attenuation in units of V-band reddened magnitudes.
            Default is `0.`.

        rv : float, optional
            Change in the reddening vector in terms of R(V)=A(V)/E(B-V).
            Default is `3.3`.

        smf : float, optional
            Secondary mass fraction for unresolved binary. Default is `0.`
            (single stellar system). Note that binaries are not permitted
            off the main sequence (`eep > 454`).

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        filt_idxs : iterable of shape `(Nfilt)`, optional
            Susbset of filter indices. If not provided, predictions in all
            filters will be returned.

        mgrid : `~numpy.ndarray` of shape `(Ngrid)`, optional
            A grid of initial mass (`mini`) that the isochrone is evaluated
            over. If not provided, the same `mini` grid that was provided
            upon initialization will be used.

        return_params : bool, optional
            Whether to return the associated parameters used to generate
            the photometry. Default is `False`.

        Returns
        -------
        iso_pred : `~numpy.ndarray` of shape `(Ngrid, Nfilt)`
            Predicted isochrone over the mass grid in magnitudes.

        params : `~numpy.ndarray` of shape `(Ngrid, Nparam)`, optional
            Parameters used to generate the isochrone.

        """

        # Initialize mass grid.
        if mgrid is None:
            mgrid = self.mgrid
        Ngrid = len(mgrid)
        params = None

        # Interpolate photometry.
        xpoints = np.c_[mgrid, np.full(Ngrid, feh), np.full(Ngrid, loga)]
        models = self.interpolator(xpoints)

        if return_params or smf != 0.:
            params = self.interpolator_p(xpoints)

        # Generate SEDs.
        if loga < loga_max:
            mags = get_seds(models[:, filt_idxs, :], av=av, rv=rv)
            mags += 5 * np.log10(dist / 1000.)  # mags relative to 1 kpc

            # Add in unresolved binaries.
            if smf != 0.:
                eepsel = np.array(params[:, -1])
                eepsel[eepsel > 454] = np.nan
                eepsel[eepsel <= 454] = 1.
                xpoints = np.c_[mgrid * smf * eepsel, np.full(Ngrid, feh),
                                np.full(Ngrid, loga)]
                models = self.interpolator(xpoints)
                mags2 = get_seds(models[:, filt_idxs, :], av=av, rv=rv)
                mags2 += 5 * np.log10(dist / 1000.)
                mags = add_mag(mags, mags2, f1=1., f2=1.)
        else:
            # Fill with nans.
            mags = np.full((Ngrid, self.mags[:, filt_idxs, :].shape[1]),
                           np.nan)

        if return_params:
            return mags, params
        else:
            return mags

    def _interpolate_sedmaker(self, loga=9., feh=0., av=0., rv=3.3, smf=0.,
                              dist=1000., loga_max=10.14,
                              filt_idxs=slice(None), mgrid=None,
                              return_params=False):
        """
        Returns a predicted isochrone for the input set of
        parameters for a specified subset of bands. Predictions
        are in apparent magnitudes at the specified distance. Based on
        interpolating over a **grid of models** to estimate EEP(age) at fixed
        mass and metallicity and subsequently generating photometry
        using a **smooth predictor**.

        Parameters
        ----------
        loga : float
            The base-10 logarithm of the isochrone age in years.
            Default is `9.`.

        feh : float, optional
            The metallicity defined logarithmically relative to the solar value.
            Default is `0.`

        av : float, optional
            Dust attenuation in units of V-band reddened magnitudes.
            Default is `0.`.

        rv : float, optional
            Change in the reddening vector in terms of R(V)=A(V)/E(B-V).
            Default is `3.3`.

        smf : float, optional
            Secondary mass fraction for unresolved binary. Default is `0.`
            (single stellar system). Note that binaries are not permitted
            off the main sequence (`eep > 454`).

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        filt_idxs : iterable of shape `(Nfilt)`, optional
            Susbset of filter indices. If not provided, predictions in all
            filters will be returned.

        mgrid : `~numpy.ndarray` of shape `(Ngrid)`, optional
            A grid of initial mass (`mini`) that the isochrone is evaluated
            over. If not provided, the same `mini` grid that was provided
            upon initialization will be used.

        return_params : bool, optional
            Whether to return the associated parameters used to generate
            the photometry. Default is `False`.

        Returns
        -------
        iso_pred : `~numpy.ndarray` of shape `(Ngrid, Nfilt)`
            Predicted isochrone over the mass grid in magnitudes.

        params : `~numpy.ndarray` of shape `(Ngrid, Nparam)`, optional
            Parameters used to generate the isochrone.

        """

        sed = self.sedmaker.payne.sed
        get_pred = self.sedmaker.get_predictions

        # Initialize mass grid.
        if mgrid is None:
            mgrid = self.mgrid
        Ngrid = len(mgrid)

        # Interpolate parameters.
        pred = ['loga', 'logt', 'logg', 'feh_surf', 'logl']
        pred_idx = [np.where(p == np.array(self.sedmaker.predictions))[0][0]
                    for p in pred]
        ageidx = pred_idx[0]
        pred_idx = pred_idx[1:]

        # Compute EEP at specific log(age) for each mass.
        eepgrid = np.arange(self.lims_eep[0], self.lims_eep[1] + 1, 1)
        Neep = len(eepgrid)
        eeps = np.zeros_like(mgrid)
        idxs = np.array([0, 1])  # bounding box for EEP
        for i, m in enumerate(mgrid):
            mconst, fconst = np.full(Neep, m), np.full(Neep, feh)
            sidx = np.arange(len(eepgrid))
            Ns = len(sidx)
            # Iterate until we find the right EEPs.
            while True:
                xpoints = np.c_[mconst[sidx[idxs]],
                                eepgrid[sidx[idxs]],
                                fconst[sidx[idxs]]]
                preds = get_pred(xpoints)
                logages = preds[:, ageidx]
                if logages[0] <= loga <= logages[1]:
                    # If we have loga within the EEPs, we're done!
                    break
                elif loga < logages[0]:
                    # Shift downwards if we're not young enough.
                    idxs -= 1
                elif loga > logages[1]:
                    # Shift upwards if we're not old enough.
                    idxs += 1
                else:
                    xs = np.c_[mconst, eepgrid, fconst]
                    sidx = np.where(np.isfinite(get_pred(xs)[:, ageidx]))[0]
                    Ns = len(sidx)
                    if len(sidx) < 2:
                        break
                    else:
                        idxs = np.array([0, 1])
                # If we're outside of the bounds, we should give up.
                if idxs[0] < 0:
                    idxs += 1
                    break
                if idxs[1] > Ns - 1:
                    idxs -= 1
                    break
            # Compute estimate of EEP.
            eeps[i] = np.interp(loga, logages, eepgrid[idxs],
                                left=np.nan, right=np.nan)

        # Compute predictions.
        xpoints = np.c_[mgrid, eeps, np.full(Ngrid, feh)]
        logt, logg, feh_surf, logl = get_pred(xpoints)[:, pred_idx].T

        # Generate SEDs.
        if loga < loga_max:
            mags = np.array([sed(logt=lt, logg=lg, feh_surf=fs, logl=ll,
                                 av=av, rv=rv, dist=dist,
                                 filt_idxs=filt_idxs)
                             for lt, lg, fs, ll in zip(logt, logg,
                                                       feh_surf, logl)])

            # Add in unresolved binaries.
            if smf != 0.:
                eeps_temp = np.array(eeps)
                eeps_temp[eeps_temp > 454] = np.nan
                xpoints = xpoints = np.c_[mgrid * smf, eeps_temp,
                                          np.full(Ngrid, feh)]
                logt, logg, feh_surf, logl = get_pred(xpoints)[:, pred_idx].T
                mags2 = np.array([sed(logt=lt, logg=lg, feh_surf=fs, logl=ll,
                                      av=av, rv=rv, dist=dist,
                                      filt_idxs=filt_idxs)
                                  for lt, lg, fs, ll in zip(logt, logg,
                                                            feh_surf, logl)])
                mags = add_mag(mags, mags2, f1=1., f2=1.)
        else:
            # Fill with nans.
            mags = np.full((Ngrid, len(self.sedmaker.filters[filt_idxs])),
                           np.nan)

        if return_params:
            return mags, np.c_[logt, logg, feh_surf, logl, eeps]
        else:
            return mags

    def get_isochrone(self, loga=9., feh=0., av=0., rv=3.3, smf=0.,
                      dist=1000., loga_max=10.14, filt_idxs=slice(None),
                      mgrid=None, return_params=False):
        """
        Returns a predicted isochrone for the input set of
        parameters for a specified subset of bands. Predictions
        are in apparent magnitudes at the specified distance.

        Parameters
        ----------
        loga : float
            The base-10 logarithm of the isochrone age in years.
            Default is `9.`.

        feh : float, optional
            The metallicity defined logarithmically relative to the solar value.
            Default is `0.`

        av : float, optional
            Dust attenuation in units of V-band reddened magnitudes.
            Default is `0.`.

        rv : float, optional
            Change in the reddening vector in terms of R(V)=A(V)/E(B-V).
            Default is `3.3`.

        smf : float, optional
            Secondary mass fraction for unresolved binary. Default is `0.`
            (single stellar system). Note that binaries are not permitted
            off the main sequence (`eep > 454`).

        dist : float, optional
            Distance in parsecs. Default is `1000.` (i.e. 1 kpc).

        loga_max : float, optional
            The maximum allowed age. No SEDs will be generated above
            `loga_max`. Default is `10.14` (13.8 Gyr).

        filt_idxs : iterable of shape `(Nfilt)`, optional
            Susbset of filter indices. If not provided, predictions in all
            filters will be returned.

        mgrid : `~numpy.ndarray` of shape `(Ngrid)`, optional
            A grid of initial mass (`mini`) that the isochrone is evaluated
            over. If not provided, the same `mini` grid that was provided
            upon initialization will be used.

        return_params : bool, optional
            Whether to return the associated parameters used to generate
            the photometry. Default is `False`.

        Returns
        -------
        iso_pred : `~numpy.ndarray` of shape `(Ngrid, Nfilt)`
            Predicted isochrone over the mass grid in magnitudes.

        params : `~numpy.ndarray` of shape `(Ngrid, Nparam)`, optional
            Parameters used to generate the isochrone.
            Includes `logt`, `logg`, `feh_surf`, `logl`, and `eep`.

        """

        if self.itype == 'sedmaker':
            return self._interpolate_sedmaker(loga=loga, feh=feh, av=av, rv=rv,
                                              smf=smf, dist=dist,
                                              loga_max=loga_max,
                                              filt_idxs=filt_idxs, mgrid=mgrid,
                                              return_params=return_params)
        elif self.itype == 'grid':
            return self._interpolate_grid(loga=loga, feh=feh, av=av, rv=rv,
                                          smf=smf, dist=dist,
                                          loga_max=loga_max,
                                          filt_idxs=filt_idxs, mgrid=mgrid,
                                          return_params=return_params)
        else:
            raise RuntimeError("Something has gone terribly wrong.")
