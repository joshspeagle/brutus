#!/usr/bin/env python
"""Rebuild the Binney & Vasiliev (2024) Milky Way model and write the tables
that brutus.priors.kinematic reads (src/brutus/priors/bv24_tables.npz).

Binney & Vasiliev (2024, MNRAS, arXiv:2306.11602; "BV24") fit one
self-consistent potential and action-based distribution functions (DFs) to
APOGEE DR17 and Gaia DR3. This script rebuilds the model with AGAMA from the
DF forms of BV24 eqs. (4)-(11) and the fitted values of BV24 Tables 3 and 4.
Values BV24 does not tabulate (the halo DFs' alpha, L0, beta and
cutoffStrength; the gas disc; the initial potential; the grids) come from
AGAMA's data/SCM_MW.ini, the configuration of Binney & Vasiliev (2023). The
rebuilt model gives K_z(R0, 1.1 kpc) / 2 pi G = 63.9 Msun/pc^2 against 66.4
in BV24 (eq. 15).

The tables, with x = R / R0 (R0 = 8.27 kpc in BV24) and z in kpc:
- dphi[i, j] = Phi(R, z) - Phi(R, 0) in (km/s)^2,
- kz[i, j] = K_z(R, z) = dPhi/dz in (km/s)^2/kpc,
on x = 0, dx, ..., X_MAX and z = Z_SCALE sinh(u) on a uniform grid in u; and
the midplane variances of the high-alpha disc, ln(sigma^2 / sigma^2(R0)) for
(R, phi, z), on x_sigma. brutus.priors.kinematic computes the thin-disk
Jeans integral from dphi and kz. vc0 is the circular speed at R0, by which
brutus scales the potential to its own v_c at the Sun.

AGAMA is a development tool only. Usage, from the repository root:
    python .github/scripts/make_bv24_tables.py /path/to/Agama/data/SCM_MW.ini
Numba caches the tables inside its compiled functions, so delete the
__pycache__ directory of src/brutus/priors after a rebuild.
"""

import configparser
import os
import sys

import agama
import numpy as np
import scipy.integrate
import scipy.optimize

R0 = 8.27
X_MAX, DX = 4.0, 0.0125
Z_MAX, Z_SCALE, NU = 100.0, 0.05, 223
X_SIGMA = np.arange(0.05, X_MAX + 1e-9, 0.025)
OUT = os.path.join("src", "brutus", "priors", "bv24_tables.npz")

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
cfg = configparser.RawConfigParser()
cfg.optionxform = str
cfg.read(sys.argv[1])


def sec(name):
    return dict(cfg[name])


def taper(E):
    """BV24 eq. (5)."""
    return E / (E + 1.0 / E)


def disc_df(M, Jphi0, Jr0, Jz0, Jint, Dint, Jext, Dext, pr, pz, Jv0, Jd0):
    """BV24 eqs. (4)-(11); a taper is off when its D is negative."""

    def df(J):
        Jr, Jz, Jp = J[:, 0], J[:, 1], J[:, 2]
        Jpp = np.maximum(Jp, 0.0)
        Jv = Jr + Jz + Jpp + Jv0
        Jd = Jr + Jz + Jpp + Jd0
        xr = (Jv / Jphi0) ** pr / Jr0
        xz = (Jv / Jphi0) ** pz / Jz0
        f = xr * np.exp(-xr * Jr) * xz * np.exp(-xz * Jz)
        f *= norm[0] / (2 * np.pi) ** 3 * Jd / Jphi0**2 * np.exp(-Jd / Jphi0)
        if Dint > 0:
            f *= taper(np.exp(np.clip((Jpp - Jint) / Dint, -300, 300)))
        if Dext > 0:
            f *= taper(np.exp(np.clip(-(Jpp - Jext) / Dext, -300, 300)))
        return np.where(Jp > 0, f, 0.0)

    norm = [1.0]
    norm[0] = M / agama.DistributionFunction(df).totalMass()
    return df


def spheroid_df(mass, J0, Jcore, Jcutoff, slopeIn, slopeOut, Fin, Fout, d):
    """The non-rotating double power law of BV23, as in AGAMA's
    py/example_self_consistent_model_mw.py."""
    alpha, L0, beta, cut = (
        float(d[k]) for k in ("alpha", "L0", "beta", "cutoffStrength")
    )

    def h_of(J):
        modJphi = abs(J[:, 2])
        L = J[:, 1] + modJphi
        c = L / (L + J[:, 0])
        jta = ((1.5 * J[:, 0] + L) / L0) ** alpha
        xi = jta / (1 + jta)
        rat = (1 - xi) * Fin + xi * Fout
        cL = np.where(
            L > 0,
            J[:, 1] * (0.5 * (rat + 1) + 0.5 * (rat - 1) * modJphi / L) + modJphi,
            0,
        )
        fac = np.exp(beta * np.sin(np.pi / 2 * c))
        return J[:, 0] / fac + 0.5 * (1 + c * xi) * fac * cL

    def df(J):
        hJ = h_of(J)
        res = (
            norm[0]
            / (2 * np.pi * J0) ** 3
            * (1 + J0 / hJ) ** slopeIn
            * (1 + hJ / J0) ** (-slopeOut)
        )
        res *= np.exp(-((hJ / Jcutoff) ** cut))
        return res * (1 + Jcore / hJ * (Jcore / hJ - zeta)) ** (-0.5 * slopeIn)

    def core(z):
        def integrand(t):
            hJ = Jcore * t * t * (3 - 2 * t) / (1 - t) ** 2 / (1 + 2 * t)
            dh = Jcore * 6 * t / (1 - t) ** 3 / (1 + 2 * t) ** 2
            return (
                hJ**2
                * dh
                * (1 + J0 / hJ) ** slopeIn
                * (1 + hJ / J0) ** (-slopeOut)
                * ((1 + Jcore / hJ * (Jcore / hJ - z)) ** (-0.5 * slopeIn) - 1)
            )

        return scipy.integrate.fixed_quad(integrand, 0.0, 1.0, n=20)[0]

    zeta = scipy.optimize.brentq(core, 0.0, 2.0)
    norm = [1.0]
    norm[0] = mass / agama.DistributionFunction(df).totalMass()
    return df


# BV24 Table 3: M (1e10 Msun), Jphi0, Jr0, Jz0, Jint, Dint, Jext, Dext, pr, pz,
# Jv0, Jd0 (actions in kpc km/s); its blank entries switch a taper off.
TABLE3 = {
    "young": (0.45, 977.9, 2.806, 1.296, 186.9, 278, 0, -1, -0.76, -0.23, 152.1, 102.1),
    "middle": (1.2, 1030, 22.82, 3.24, 0, -1, 0, -1, -0.23, -0.7, 146.4, 731.9),
    "old": (0.87, 508, 47.14, 24.04, 0, -1, 0, -1, 0.034, -0.043, 132.7, 733.9),
    "highA": (0.766, 399, 116.4, 64.6, 0, -1, 2212, 207.8, 0.1, 0.17, 150, 40),
    "bulge": (1.27, 127.5, 122.2, 34.21, 0, -1, 611.2, 217.5, 0.82, -0.13, 150, 20),
}
discs = {k: disc_df(v[0] * 1e10, *v[1:]) for k, v in TABLE3.items()}
# BV24 Table 4: M, J0, Jcore, Jcutoff, alpha_in, alpha_out, F_in, F_out.
dark = spheroid_df(94e10, 1e4, 100, 2e4, 1.6, 2.7, 1.4, 1.2, sec("DF dark halo"))
stellar_halo = spheroid_df(
    0.03969e10, 583.2, 6.148, 1e5, 1.6, 4.2, 1.8, 1.2, sec("DF stellar halo")
)

model = agama.SelfConsistentModel(**sec("SelfConsistentModel"))
model.potential = agama.Potential(
    *[
        sec("Potential " + n)
        for n in ("thin disk", "thick disk", "gas disk", "dark halo", "bulge")
    ]
)
stellar = agama.DistributionFunction(*discs.values(), stellar_halo)
model.components = [
    agama.Component(df=stellar, disklike=True, **sec("SelfConsistentModel disk")),
    agama.Component(df=dark, disklike=False, **sec("SelfConsistentModel halo")),
    agama.Component(density=agama.Density(**sec("Potential gas disk")), disklike=True),
]
with agama.setNumThreads(4):
    for _ in range(5):
        model.iterate()
pot = model.potential
G2 = 2 * np.pi * 4.30091e-6  # 2 pi G in kpc (km/s)^2 / Msun
print(
    "K_z(R0, 1.1 kpc) / 2 pi G = %.1f Msun/pc^2"
    % (-pot.force(R0, 0, 1.1)[2] / G2 * 1e-6)
)
vc0 = np.sqrt(-R0 * pot.force(R0, 0, 0)[0])
print("v_c(R0) = %.2f km/s" % vc0)

# Vertical potential and K_z.
x = np.arange(0.0, X_MAX + DX / 2, DX)
u = np.linspace(0.0, np.arcsinh(Z_MAX / Z_SCALE), NU)
z = Z_SCALE * np.sinh(u)
R = np.maximum(x * R0, 1e-3)


def at(Rs, zs):
    return np.column_stack((Rs.ravel(), 0 * Rs.ravel(), zs.ravel()))


Rg, Zg = np.meshgrid(R, z, indexing="ij")
dphi = pot.potential(at(Rg, Zg)).reshape(Rg.shape)
dphi -= dphi[:, :1]
kz = -pot.force(at(Rg, Zg))[:, 2].reshape(Rg.shape)
# Midplane dispersions of the high-alpha disc.
gm = agama.GalaxyModel(pot, discs["highA"])
Rs = X_SIGMA * R0
with agama.setNumThreads(4):
    _, mv, mv2 = gm.moments(at(Rs, 0 * Rs), dens=True, vel=True, vel2=True)
var = np.column_stack((mv2[:, 0], mv2[:, 1] - mv[:, 1] ** 2, mv2[:, 2]))
ln_var = np.log(var) - np.log([np.interp(1.0, X_SIGMA, v) for v in var.T])

np.savez(
    OUT,
    vc0=vc0,
    dx=DX,
    z_scale=Z_SCALE,
    du=u[1] - u[0],
    dphi=dphi,
    kz=kz,
    x_sigma=X_SIGMA,
    ln_var=ln_var,
)
print("wrote", OUT, dphi.shape, ln_var.shape)
