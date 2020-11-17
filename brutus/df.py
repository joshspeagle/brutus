#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF functions.

"""

from __future__ import (print_function, division)

import numpy as np
from astropy import units as u
from astropy.constants import G
from galpy.potential import DoubleExponentialDiskPotential, HernquistPotential
from galpy.potential import evaluateDensities, evaluateRforces, evaluatezforces
from galpy.potential import evaluateR2derivs, evaluateRzderivs
from scipy.integrate import cumtrapz
import pickle
import time

__all__ = []

def _save_force_grid(fname, force_grid):
    pickle.dump(force_grid, open(fname, 'wb'))

def _load_force_grid(fname):
    return pickle.load(force_grid, open(fname, 'rb'))

def _init_vel(R_solar=8.2, Z_solar=0.025, R_thin=2.6, Z_thin=0.3,
                 R_thick=2.0, Z_thick=0.9, fgas=0.1, Z_gas=0.15,
                 Mthin=3.5E10, Mthick=6E9,
                 vcirc_solar=220, RadialDispersionFactor=1.7,
                 Mhalo=1E12, Rhalo=26,
                 ro=8, vo=220,
                 RSIZE=512, ZSIZE=512, RMIN=0.002, ZMIN=0.002, RMAX=200, ZMAX=200):

    # Get G.
    G_val = G.to_value(ro * u.kpc *(vo * u.km/u.s)**2 / u.Msun)

    # First generate the force grid.
    force_grid = {}

    Rlist = np.zeros(RSIZE)
    Rlist[1:] = np.logspace(np.log10(RMIN), np.log10(RMAX), RSIZE-1)

    zlist = np.zeros(ZSIZE)
    zlist[1:] = np.logspace(np.log10(ZMIN), np.log10(ZMAX), ZSIZE-1)

    force_grid['Rlist'] = Rlist
    force_grid['zlist'] = zlist

    force_grid['R'], force_grid['z'] = np.meshgrid(Rlist, zlist, indexing='ij')

    # Thin disk potential.
    # amp_thin = Mthin / (4. *np.pi * R_thin**2 * Z_thin) * (u.Msun/u.kpc**3)
    amp_thin = Mthin / (4. *np.pi * R_thin**2 * Z_thin)
    amp_thin *= (u.Msun/u.kpc**3)
    pot_thin = DoubleExponentialDiskPotential(amp=amp_thin, hr=R_thin * u.kpc,
                                              hz=Z_thin * u.kpc, ro=ro*u.kpc, vo=vo*u.km/u.s)
    
    # amp_thick = Mthick / (4. *np.pi * R_thick**2 * Z_thick) * (u.Msun/u.kpc**3)
    amp_thick = Mthick / (4. *np.pi * R_thick**2 * Z_thick)
    amp_thick *= (u.Msun/u.kpc**3)
    pot_thick = DoubleExponentialDiskPotential(amp=amp_thick, hr=R_thick * u.kpc,
                                              hz=Z_thick * u.kpc,
                                              ro=ro * u.kpc, vo=vo * u.km/u.s)
    
    amp_halo = 2. * Mhalo 
    amp_halo *= u.Msun
    pot_halo = HernquistPotential(amp=amp_halo, a=Rhalo * u.kpc,
                                  ro=ro * u.kpc, vo=vo * u.km/u.s)
    
    pot = [pot_thin, pot_thick, pot_halo]

    t0 = time.time()
    # Now compute densities and forces.

    force_grid['Density_thin']   = evaluateDensities(pot_thin, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in Msun/pc^3
    force_grid['Density_thick']  = evaluateDensities(pot_thick, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in Msun/pc^3
    force_grid['Density_thin']  *= 1E9 # convert to Msun/kpc^3
    force_grid['Density_thick'] *= 1E9 # convert to Msun/kpc^3


    # NOTE: these functions are not general and depend on the radial surface density profile being exponential
    force_grid['DR_Density_thin']  =  (-1.0/R_thin) * force_grid['Density_thin'] # in Msun/kpc^2
    force_grid['DR_Density_thick'] = (-1.0/R_thick) * force_grid['Density_thick'] # in Msun/kpc^2

    force_grid['DPhi_R'] = -evaluateRforces(pot, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc)  # in km/s/Myr
    force_grid['DPhi_z'] = -evaluatezforces(pot, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc)  # in km/s/Myr
    force_grid['DPhi_R'] *= 977.79222167 # convert to (km/s)^2 / kpc
    force_grid['DPhi_z'] *= 977.79222167 # convert to (km/s)^2 / kpc

    force_grid['DPhi2_R']   = evaluateR2derivs(pot, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in 1/Gyr^2
    force_grid['DPhi2_Rz']  = evaluateRzderivs(pot, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in 1/Gyr^2
    force_grid['DPhi2_R']  *= 0.95607763 # convert to (km/s)^2 / kpc^2
    force_grid['DPhi2_Rz'] *= 0.95607763 # convert to (km/s)^2 / kpc^2

    print('force omputation took :', time.time()-t0)

    # Now compute velocity moments.
    force_grid = _do_jeans_modelling(force_grid, RadialDispersionFactor, ro, vo)

    return force_grid

def _do_jeans_modelling(force_grid, RadialDispersionFactor, ro, vo):
    rho_thin, rho_thick = force_grid['Density_thin'], force_grid['Density_thick']
    DR_rho_thin, DR_rho_thick = force_grid['DR_Density_thin'], force_grid['DR_Density_thick']
    DPhi_R, DPhi2_R, DPhi_z, DPhi2_Rz = force_grid['DPhi_R'], force_grid['DPhi2_R'], force_grid['DPhi_z'], force_grid['DPhi2_Rz']
    Rlist, zlist = force_grid['Rlist'], force_grid['zlist']
    R, z = force_grid['R'], force_grid['z']

    # Rlist *= u.kpc
    # zlist *= u.kpc
    # R *= u.kpc
    # z *= u.kpc

    # First compute integral of rho * force
    integrand = rho_thin * DPhi_z
    VelDispRz_thin = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    VelDispRz_thin = np.transpose((VelDispRz_thin[:,-1] - np.transpose(VelDispRz_thin)))
    VelDispRz_thin /= rho_thin
    VelDispRz_thin[np.isnan(VelDispRz_thin)] = 0.0  # Fixes where rho is 0

    integrand = rho_thick * DPhi_z
    VelDispRz_thick = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    VelDispRz_thick = np.transpose((VelDispRz_thick[:,-1] - np.transpose(VelDispRz_thick)))
    VelDispRz_thick /= rho_thick
    VelDispRz_thick[np.isnan(VelDispRz_thick)] = 0.0  # Fixes where rho is 0

    # Now compute the derivative wrt R
    integrand = DR_rho_thin * DPhi_z + rho_thin * DPhi2_Rz
    DR_RhoVelDispRz_thin = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    DR_RhoVelDispRz_thin = np.transpose((DR_RhoVelDispRz_thin[:,-1] - np.transpose(DR_RhoVelDispRz_thin)))

    integrand = DR_rho_thick * DPhi_z + rho_thick * DPhi2_Rz
    DR_RhoVelDispRz_thick = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    DR_RhoVelDispRz_thick = np.transpose((DR_RhoVelDispRz_thick[:,-1] - np.transpose(DR_RhoVelDispRz_thick)))

    # Put it all together
    VCircsq = R * DPhi_R
    epi_gamma2 = 1./((3./R)*DPhi_R + DPhi2_R)
    epi_gamma2 *= (4./R) * DPhi_R
    epi_gamma2 = epi_gamma2

    # Here, VelDispPhi is actually avg(v^2), not std(v)^2
    VelDispPhi_thin = VelDispRz_thin + (R/rho_thin) * DR_RhoVelDispRz_thin + VCircsq
    VelDispPhi_thick = VelDispRz_thick + (R/rho_thick) * DR_RhoVelDispRz_thick + VCircsq

    VelStreamPhi_thin = VelDispPhi_thin - RadialDispersionFactor * VelDispRz_thin/epi_gamma2
    VelStreamPhi_thick = VelDispPhi_thick - RadialDispersionFactor * VelDispRz_thick/epi_gamma2

    # Now, we correctly set vel disper
    # print(VelDispPhi_thin.unit)
    # print(VelStreamPhi_thin.unit)
    print(np.shape(VelDispPhi_thick))
    print(np.shape(VelStreamPhi_thick))

    VelDispPhi_thin = VelDispPhi_thin - VelStreamPhi_thin
    VelDispPhi_thick = VelDispPhi_thick - VelStreamPhi_thick

    VelStreamPhi_thin = np.sqrt(VelStreamPhi_thin)
    VelStreamPhi_thick = np.sqrt(VelStreamPhi_thick)

    # Now set other streaming velocities
    VelStreamR_thin = np.zeros(np.shape(VelStreamPhi_thin))
    VelStreamz_thin = np.zeros(np.shape(VelStreamPhi_thin))
    VelStreamR_thick = np.zeros(np.shape(VelStreamPhi_thick))
    VelStreamz_thick = np.zeros(np.shape(VelStreamPhi_thick))

    # Now time to get back on the grid.
    force_grid['VCircsq'] = VCircsq
    force_grid['VCircsq'][0][0] = 0.0 # fix nan
    force_grid['epi_gamma2'] = epi_gamma2

    force_grid['VelStreamR_thin'] = VelStreamR_thin
    force_grid['VelStreamPhi_thin'] = VelStreamPhi_thin
    force_grid['VelStreamz_thin'] = VelStreamz_thin
    force_grid['VelDispR_thin'] = RadialDispersionFactor * VelDispRz_thin
    force_grid['VelDispPhi_thin'] = VelDispPhi_thin
    force_grid['VelDispz_thin'] = VelDispRz_thin

    force_grid['VelStreamR_thick'] = VelStreamR_thick
    force_grid['VelStreamPhi_thick'] = VelStreamPhi_thick
    force_grid['VelStreamz_thick'] = VelStreamz_thick
    force_grid['VelDispR_thick'] = RadialDispersionFactor * VelDispRz_thick
    force_grid['VelDispPhi_thick'] = VelDispPhi_thick
    force_grid['VelDispz_thick'] = VelDispRz_thick

    return force_grid


if __name__ == '__main__':
    force_grid = _init_vel(RSIZE=16)

