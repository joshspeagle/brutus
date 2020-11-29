#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF functions.

"""

from __future__ import (print_function, division)

import numpy as np
from astropy import units as u
from astropy.constants import G
from galpy.potential import DoubleExponentialDiskPotential, HernquistPotential, TwoPowerSphericalPotential
from galpy.potential import evaluateDensities, evaluateRforces, evaluatezforces
from galpy.potential import evaluateR2derivs, evaluateRzderivs
from scipy.integrate import cumtrapz
from scipy.interpolate import RectBivariateSpline
import pickle
import time

__all__ = []

# TODO: 
# - do radial derivatives of density numerically instead of analytically

def _save_force_grid(fname, force_grid):
    pickle.dump(force_grid, open(fname, 'wb'))

def _load_force_grid(fname):
    return pickle.load(force_grid, open(fname, 'rb'))

def _init_vel(R_solar=8.2, Z_solar=0.025, R_thin=2.6, Z_thin=0.3,
                 R_thick=2.0, Z_thick=0.9, fgas=0.1, Z_gas=0.15,
                 Mthin=3.5E10, Mthick=6E9,
                 Mstarhalo=5E8, Rs=25, alpha_in=2.5, alpha_out=4.0,
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
    
    amp_star_halo = Mstarhalo
    amp_star_halo *= u.Msun
    pot_star_halo = TwoPowerSphericalPotential(amp=amp_star_halo, a=Rs * u.kpc, alpha=alpha_in,
                                               beta=alpha_out, ro=ro * u.kpc, vo = vo*u.km/u.s)
    
    amp_halo = 2. * Mhalo 
    amp_halo *= u.Msun
    pot_halo = HernquistPotential(amp=amp_halo, a=Rhalo * u.kpc,
                                  ro=ro * u.kpc, vo=vo * u.km/u.s)
    
    pot = [pot_thin, pot_thick, pot_star_halo, pot_halo]

    t0 = time.time()
    # Now compute densities and forces.

    force_grid['Density_thin']   = evaluateDensities(pot_thin, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in Msun/pc^3
    force_grid['Density_thick']  = evaluateDensities(pot_thick, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in Msun/pc^3
    force_grid['Density_star_halo']  = evaluateDensities(pot_star_halo, force_grid['R'] * u.kpc, force_grid['z'] * u.kpc) # in Msun/pc^3
    force_grid['Density_thin']  *= 1E9 # convert to Msun/kpc^3
    force_grid['Density_thick'] *= 1E9 # convert to Msun/kpc^3
    force_grid['Density_star_halo'] *= 1E9 # convert to Msun/kpc^3

    # set density star halo at R=z=0 to be a really big number since the density is formally infinite, but this messes up stuff later
    force_grid['Density_star_halo'][0][0] = 1E99

    # NOTE: these functions are not general and depend on the radial surface density profile being exponential
    force_grid['DR_Density_thin']  =  (-1.0/R_thin) * force_grid['Density_thin'] # in Msun/kpc^2
    force_grid['DR_Density_thick'] = (-1.0/R_thick) * force_grid['Density_thick'] # in Msun/kpc^2

    # NOTE: This one depends on the two power spherical potential...
    r = np.sqrt(force_grid['R']**2 + force_grid['z']**2)
    deriv = ((alpha_in - alpha_out)/Rs) * force_grid['Density_star_halo'] / (1 + r/Rs)
    deriv -= (alpha_in/Rs) * force_grid['Density_star_halo'] / (r/Rs)
    force_grid['DR_Density_star_halo'] = (force_grid['R']/r) * deriv

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

    # Now construct splines.
    force_grid = _construct_spline(force_grid)

    return force_grid

def _do_jeans_modelling(force_grid, RadialDispersionFactor, ro, vo):
    rho_thin, rho_thick, rho_halo = force_grid['Density_thin'], force_grid['Density_thick'], force_grid['Density_star_halo']
    DR_rho_thin, DR_rho_thick, DR_rho_halo = force_grid['DR_Density_thin'], force_grid['DR_Density_thick'], force_grid['DR_Density_star_halo']
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

    integrand = rho_halo * DPhi_z
    VelDispRz_halo = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    VelDispRz_halo = np.transpose((VelDispRz_halo[:,-1] - np.transpose(VelDispRz_halo)))
    VelDispRz_halo /= rho_halo
    VelDispRz_halo[np.isnan(VelDispRz_halo)] = 0.0  # Fixes where rho is 0

    # Now compute the derivative wrt R
    integrand = DR_rho_thin * DPhi_z + rho_thin * DPhi2_Rz
    DR_RhoVelDispRz_thin = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    DR_RhoVelDispRz_thin = np.transpose((DR_RhoVelDispRz_thin[:,-1] - np.transpose(DR_RhoVelDispRz_thin)))

    integrand = DR_rho_thick * DPhi_z + rho_thick * DPhi2_Rz
    DR_RhoVelDispRz_thick = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    DR_RhoVelDispRz_thick = np.transpose((DR_RhoVelDispRz_thick[:,-1] - np.transpose(DR_RhoVelDispRz_thick)))

    integrand = DR_rho_halo * DPhi_z + rho_halo * DPhi2_Rz
    DR_RhoVelDispRz_halo = cumtrapz(integrand, zlist/ro, initial=0, axis=1)
    DR_RhoVelDispRz_halo = np.transpose((DR_RhoVelDispRz_halo[:,-1] - np.transpose(DR_RhoVelDispRz_halo)))

    DR_RhoVelDispRz_thin[0,:] = 0.0 # unstable at R=0
    DR_RhoVelDispRz_thick[0,:] = 0.0 # unstable at R=0
    DR_RhoVelDispRz_halo[0,:] = 0.0 # unstable at R=0

    # Put it all together
    VCircsq = R * DPhi_R
    epi_gamma2 = 1./((3./R)*DPhi_R + DPhi2_R)
    epi_gamma2 *= (4./R) * DPhi_R
    epi_gamma2 = epi_gamma2

    epi_gamma2[0,:] = 0.0 # at R=0 above gives nans
    VCircsq[0,0] = 0.0 # similar

    # VCircsq[np.isnan(VCircsq)] = 0.0
    # epi_gamma2[np.isnan(epi_gamma2)] = 0.0

    # Here, VelDispPhi is actually avg(v^2), not std(v)^2
    VelDispPhi_thin = VelDispRz_thin + (R/rho_thin) * DR_RhoVelDispRz_thin + VCircsq
    VelDispPhi_thick = VelDispRz_thick + (R/rho_thick) * DR_RhoVelDispRz_thick + VCircsq
    VelDispPhi_halo = VelDispRz_halo + (R/rho_halo) * DR_RhoVelDispRz_halo + VCircsq

    # print('ISNAN=', len(np.where(np.isnan(epi_gamma2))[0]))
    # print('ISNAN=', len(np.where(np.isnan((R/rho_thin)))[0]))
    # print('ISNAN=', len(np.where(np.isnan(( DR_RhoVelDispRz_thin)))[0]))
    # print('ISNAN=', len(np.where(np.isnan(VCircsq))[0]))

    # print(np.where(np.isnan(DR_RhoVelDispRz_thin))[0])
    # print(np.where(np.isnan(DR_RhoVelDispRz_thin))[1])

    # print(np.where(np.isnan(VelDispPhi_thin))[0])
    # print(np.where(np.isnan(VelDispPhi_thin))[1])

    # VelDispPhi_thin[rho_thin < 1e-12] = 0.0
    # VelDispPhi_thick[rho_thick < 1e-12] = 0.0


    VelStreamPhi_thin = VelDispPhi_thin - RadialDispersionFactor * VelDispRz_thin/epi_gamma2
    VelStreamPhi_thick = VelDispPhi_thick - RadialDispersionFactor * VelDispRz_thick/epi_gamma2
    VelStreamPhi_halo = np.zeros(np.shape(VelDispPhi_halo))

    # Now, we correctly set vel disper
    # print(VelDispPhi_thin.unit)
    # print(VelStreamPhi_thin.unit)
    print(np.shape(VelDispPhi_thick))
    print(np.shape(VelStreamPhi_thick))

    VelDispPhi_thin = VelDispPhi_thin - VelStreamPhi_thin
    VelDispPhi_thick = VelDispPhi_thick - VelStreamPhi_thick


    # fix nans and negatives
    VelStreamPhi_thin[VelStreamPhi_thin < 0.0] = 0.0
    VelStreamPhi_thick[VelStreamPhi_thick < 0.0] = 0.0
    VelStreamPhi_halo[VelStreamPhi_halo < 0.0] = 0.0
    VelStreamPhi_thin[np.isnan(VelStreamPhi_thin)] = 0.0
    VelStreamPhi_thick[np.isnan(VelStreamPhi_thick)] = 0.0
    VelStreamPhi_halo[np.isnan(VelStreamPhi_halo)] = 0.0

    VelStreamPhi_thin = np.sqrt(VelStreamPhi_thin)
    VelStreamPhi_thick = np.sqrt(VelStreamPhi_thick)
    VelStreamPhi_halo = np.sqrt(VelStreamPhi_halo)

    VelDispPhi_thin[np.isnan(VelDispPhi_thin)] = 0.0
    VelDispPhi_thick[np.isnan(VelDispPhi_thick)] = 0.0
    VelDispPhi_halo[np.isnan(VelDispPhi_halo)] = 0.0

    # Now set other streaming velocities
    VelStreamR_thin = np.zeros(np.shape(VelStreamPhi_thin))
    VelStreamz_thin = np.zeros(np.shape(VelStreamPhi_thin))
    VelStreamR_thick = np.zeros(np.shape(VelStreamPhi_thick))
    VelStreamz_thick = np.zeros(np.shape(VelStreamPhi_thick))
    VelStreamR_halo = np.zeros(np.shape(VelStreamPhi_halo))
    VelStreamz_halo = np.zeros(np.shape(VelStreamPhi_halo))

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

    force_grid['VelStreamR_halo'] = VelStreamR_halo
    force_grid['VelStreamPhi_halo'] = VelStreamPhi_halo
    force_grid['VelStreamz_halo'] = VelStreamz_halo
    force_grid['VelDispR_halo'] = RadialDispersionFactor * VelDispRz_halo
    force_grid['VelDispPhi_halo'] = VelDispPhi_halo
    force_grid['VelDispz_halo'] = VelDispRz_halo

    return force_grid

def _construct_spline(force_grid):
    for key in ['Density_thin', 'Density_thick', 'Density_star_halo',
                'VelStreamR_thin', 'VelStreamPhi_thin', 'VelStreamz_thin',
                'VelDispR_thin', 'VelDispPhi_thin', 'VelDispz_thin' ,
                'VelStreamR_thick', 'VelStreamPhi_thick', 'VelStreamz_thick',
                'VelDispR_thick', 'VelDispPhi_thick', 'VelDispz_thick',
                'VelStreamR_halo', 'VelStreamPhi_halo', 'VelStreamz_halo',
                'VelDispR_halo', 'VelDispPhi_halo', 'VelDispz_halo']:
        force_grid['spline_'+key] = RectBivariateSpline(force_grid['Rlist'], force_grid['zlist'], force_grid[key])

    return force_grid

def get_velocity_ellipsoid(R, z, force_grid):
    N = len(R)
    assert N==len(z), "R and z must have the same length"
    
    ans = {}
    for key in ['Density_thin', 'Density_thick', 'Density_star_halo',
                'VelStreamR_thin', 'VelStreamPhi_thin', 'VelStreamz_thin',
                'VelDispR_thin', 'VelDispPhi_thin', 'VelDispz_thin' ,
                'VelStreamR_thick', 'VelStreamPhi_thick', 'VelStreamz_thick',
                'VelDispR_thick', 'VelDispPhi_thick', 'VelDispz_thick',
                'VelStreamR_halo', 'VelStreamPhi_halo', 'VelStreamz_halo',
                'VelDispR_halo', 'VelDispPhi_halo', 'VelDispz_halo']:
        ans[key] = force_grid['spline_'+key](R, z, grid=False)
    
    # now determine the prob of being in each component based on density.
    tot_den = ans['Density_thin'] + ans['Density_thick'] + ans['Density_star_halo']
    fthin = ans['Density_thin'] / tot_den
    fthick = ans['Density_thick'] / tot_den
    fhalo = ans['Density_star_halo'] / tot_den
    fcomp = np.transpose([fthin, fthick, fhalo])

    # now put together the mean matrix
    mean_vR_thin, mean_vPhi_thin, mean_vz_thin = ans['VelStreamR_thin'], ans['VelStreamPhi_thin'], ans['VelStreamz_thin']
    mean_vR_thick, mean_vPhi_thick, mean_vz_thick = ans['VelStreamR_thick'], ans['VelStreamPhi_thick'], ans['VelStreamz_thick']
    mean_vR_halo, mean_vPhi_halo, mean_vz_halo = ans['VelStreamR_halo'], ans['VelStreamPhi_halo'], ans['VelStreamz_halo']

    mean_mat = np.array([[mean_vR_thin, mean_vPhi_thin, mean_vz_thin], 
                         [mean_vR_thick, mean_vPhi_thick, mean_vz_thick],
                         [mean_vR_halo, mean_vPhi_halo, mean_vz_halo]])

    # and put together the covariance matrix
    # for now, it is assumed that the off-diagonal terms are zero, but they are included for now
    # so that this assumption can be changed in the future.
    disp_vR_thin, disp_vPhi_thin, disp_vz_thin = ans['VelDispR_thin'], ans['VelDispPhi_thin'], ans['VelDispz_thin']
    disp_vR_thick, disp_vPhi_thick, disp_vz_thick = ans['VelDispR_thick'], ans['VelDispPhi_thick'], ans['VelDispz_thick']
    disp_vR_halo, disp_vPhi_halo, disp_vz_halo = ans['VelDispR_halo'], ans['VelDispPhi_halo'], ans['VelDispz_halo']

    # disp_vR_thin, disp_vPhi_thin, disp_vz_thin = np.random.rand(N), np.random.rand(N), np.random.rand(N)
    # disp_vR_thick, disp_vPhi_thick, disp_vz_thick = np.random.rand(N), np.random.rand(N), np.random.rand(N)

    disp_vRPhi_thin, disp_vPhiz_thin, disp_vzR_thin = np.zeros(N), np.zeros(N), np.zeros(N)
    disp_vRPhi_thick, disp_vPhiz_thick, disp_vzR_thick = np.zeros(N), np.zeros(N), np.zeros(N)
    disp_vRPhi_halo, disp_vPhiz_halo, disp_vzR_halo = np.zeros(N), np.zeros(N), np.zeros(N)

    cov_thin = np.array([[disp_vR_thin, disp_vRPhi_thin, disp_vzR_thin],
                         [disp_vRPhi_thin, disp_vPhi_thin, disp_vPhiz_thin],
                         [disp_vzR_thin, disp_vPhiz_thin, disp_vz_thin]])
    
    cov_thick = np.array([[disp_vR_thick, disp_vRPhi_thick, disp_vzR_thick],
                          [disp_vRPhi_thick, disp_vPhi_thick, disp_vPhiz_thick],
                          [disp_vzR_thick, disp_vPhiz_thick, disp_vz_thick]])
    
    cov_halo = np.array([[disp_vR_halo, disp_vRPhi_halo, disp_vzR_halo],
                          [disp_vRPhi_halo, disp_vPhi_halo, disp_vPhiz_halo],
                          [disp_vzR_halo, disp_vPhiz_halo, disp_vz_halo]])
    
    cov = np.array([cov_thin, cov_thick, cov_halo])

    return fcomp, mean_mat, cov



if __name__ == '__main__':
    force_grid = _init_vel(RSIZE=16)
    # force_grid = _init_vel(RSIZE=512)

    fcomp, mean_mat, cov = get_velocity_ellipsoid(10.*np.random.rand(100), 10.*np.random.rand(100), force_grid)

