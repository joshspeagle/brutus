"""Tests of the kinematic term of the Galactic prior and the bulge velocities."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from brutus.priors import galactic as gal
from brutus.priors import kinematic as kin
from brutus.priors import logp_galactic_structure
from brutus.priors.kinematic import (
    bulge_velocity_moments,
    halo_velocity_moments,
    thick_disk_velocity_moments,
    thin_disk_velocity_moments,
    velocity_to_observables,
)
from brutus.utils.math import R_SOLAR, galactic_to_galactocentric_cyl

COORD = (30.0, 5.0)
PM = np.array([-4.0, -1.0])
PM_COV = np.array([[0.05, 0.01], [0.01, 0.04]])
FEH, AGE = -0.4, 4.0


def _prior(d, **kwargs):
    n = len(d)
    return logp_galactic_structure(
        d,
        COORD,
        feh=np.full(n, FEH),
        loga=np.full(n, np.log10(AGE * 1e9)),
        return_components=True,
        **kwargs,
    )


def _obs(rv):
    """The observed motion and its covariance, with or without a radial
    velocity."""
    if rv is None:
        return PM, PM_COV
    cov = np.zeros((3, 3))
    cov[:2, :2] = PM_COV
    cov[2, 2] = rv[1] ** 2
    return np.r_[PM, rv[0]], cov


def _ln_gauss(d, mean, cov, rv, outlier):
    """ln of the outlier mixture of the observed motion under the
    ellipsoids (mean, cov) at the distances d, with velocity_to_observables
    as the reference projection."""
    obs, obs_cov = _obs(rv)
    n = len(obs)
    om, oc = velocity_to_observables(d, COORD, mean, cov)
    var = gal.PM_OUTLIER_SIGMA**2
    zero = np.zeros((len(d), 3))
    um, uc = velocity_to_observables(
        d, COORD, zero, var * np.eye(3)[None] * np.ones((len(d), 1, 1))
    )
    out = []
    for i in range(len(d)):
        ln_c = multivariate_normal.logpdf(obs, om[i, :n], oc[i, :n, :n] + obs_cov)
        ln_u = multivariate_normal.logpdf(obs, um[i, :n], uc[i, :n, :n] + obs_cov)
        if outlier == 0:
            out.append(ln_c)
        else:
            out.append(np.logaddexp(np.log1p(-outlier) + ln_c, np.log(outlier) + ln_u))
    return np.array(out)


@pytest.mark.parametrize("rv", [None, (20.0, 2.0)])
@pytest.mark.parametrize("outlier", [0.0, 0.01])
def test_kinematic_term_of_thick_disk_halo_and_bulge_matches_the_projected_ellipsoids(
    rv, outlier
):
    # 7.5 kpc toward (30, 5) lies inside the bulge.
    d = np.array([0.5, 2.0, 7.5, 12.0])
    extra = {} if rv is None else {"rv": rv[0], "rv_err": rv[1]}
    _, comps = _prior(d, pm=PM, pm_cov=PM_COV, pm_outlier=outlier, **extra)
    R, Z = galactic_to_galactocentric_cyl(d, *COORD)[:2]
    thick = thick_disk_velocity_moments(R, R_scale=1.9, R_smooth=2.0)
    np.testing.assert_allclose(
        comps["kinematics"][1], _ln_gauss(d, *thick, rv, outlier), rtol=1e-9
    )
    # The halo is a mixture of three ellipsoids, each mixed with the outlier.
    w, m, c = halo_velocity_moments(R, Z)
    lobes = [
        np.log(w[:, k]) + _ln_gauss(d, m[:, k], c[:, k], rv, 0.0) for k in range(3)
    ]
    ln_halo = np.logaddexp.reduce(lobes, axis=0)
    if outlier:
        var = gal.PM_OUTLIER_SIGMA**2
        cov = var * np.eye(3)[None] * np.ones((len(d), 1, 1))
        obs_out = _ln_gauss(d, np.zeros((len(d), 3)), cov, rv, 0.0)
        ln_halo = np.logaddexp(np.log1p(-outlier) + ln_halo, np.log(outlier) + obs_out)
    np.testing.assert_allclose(comps["kinematics"][2], ln_halo, rtol=1e-9)
    inside = comps["number_density"][3] > gal.LOG_ZERO / 2
    assert inside.tolist() == [False, False, True, False]
    bulge = bulge_velocity_moments(R[inside], Z[inside])
    np.testing.assert_allclose(
        comps["kinematics"][3][inside],
        _ln_gauss(d[inside], *bulge, rv, outlier),
        rtol=1e-9,
    )


def test_kinematic_term_of_the_thin_disk_sums_the_ellipsoid_of_each_true_age():
    # Each true-age node of the labelled thin disk takes the ellipsoid of its
    # own age, computed here with thin_disk_velocity_moments, and the node sum
    # weighs them as the density does.
    d = np.array([0.5, 2.0, 7.5])
    _, comps = _prior(d, pm=PM, pm_cov=PM_COV)
    D = gal._NORMALIZATION_DEFAULTS
    disk, _, tables, _, _ = gal._normalizations(**D)
    f_L, _, ln_norm, ln_mass = tables
    labels, thick_ages, halo_feh, bar_pops = gal._label_args(
        (-0.45, 0.22, -0.5),
        (13.0, -4.0, 0.8),
        (-1.18, 0.28, -0.67, -1.95, 1.0),
        (12.0, 1.5),
        (0.6, -0.31, 0.31, 0.26, 0.20),
        (11.0, 1.5, 8.0, 3.0),
        D["feh_systematic"],
        (0.1, 0.15, 0.15),
        0.2,
    )
    terms, _ = gal._label_terms(
        FEH,
        AGE,
        True,
        True,
        labels,
        thick_ages,
        halo_feh,
        bar_pops,
        np.empty((0, 0, 0)),
        np.empty((0, gal._NODE_COLUMNS)),
        np.empty(0),
    )
    nodes = terms[3]
    ages = nodes[:, -1]
    R, Z = galactic_to_galactocentric_cyl(d, *COORD)[:2]
    for i in range(len(d)):
        n = len(ages)
        mean, cov = thin_disk_velocity_moments(
            np.full(n, R[i]), np.full(n, Z[i]), ages, np.full(n, FEH)
        )
        extra = _ln_gauss(np.full(n, d[i]), mean, cov, None, gal.PM_OUTLIER)
        x = gal._soft_clamp(R[i], gal.DISK_INNER_RADIUS, np.inf, gal.DISK_EDGE)[0]
        x /= R_SOLAR
        ln_fz2, jeans, _ = kin._vertical_terms(x, abs(Z[i]))
        R_feh, R_age, _ = gal._thin_disk_fit_radius(x * R_SOLAR, ln_norm)
        args = (nodes, R_feh, R_age, x, disk[1], f_L, ln_fz2, jeans, ln_mass)
        ref = gal._thin_disk_ln_feh_sum(*args, extra)
        ref -= gal._thin_disk_ln_feh_sum(*args, np.empty(0))
        assert comps["kinematics"][0][i] == pytest.approx(ref, abs=1e-9)


def test_motion_terms_add_to_the_prior_and_switched_off_components_take_none():
    d = np.array([0.5, 2.0, 7.5, 12.0])
    lp, comps = _prior(d, pm=PM, pm_cov=PM_COV, bar_mass=0.0)
    total = np.logaddexp.reduce(
        [
            comps["number_density"][k] + comps["labels"][k] + comps["kinematics"][k]
            for k in range(3)
        ],
        axis=0,
    )
    np.testing.assert_allclose(lp, total, rtol=1e-12)
    assert np.all(comps["number_density"][3] <= gal.LOG_ZERO / 2)
    np.testing.assert_array_equal(comps["kinematics"][3], 0.0)


def test_proper_motion_floor_adds_to_the_proper_motion_variances():
    d = np.array([0.8, 3.0])
    floor = 0.3
    with_floor = _prior(d, pm=PM, pm_cov=PM_COV, pm_floor=floor)[0]
    widened = _prior(d, pm=PM, pm_cov=PM_COV + floor**2 * np.eye(2))[0]
    np.testing.assert_allclose(with_floor, widened, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pm": PM}, "pm_cov"),
        ({"pm": PM, "pm_cov": np.eye(3)}, "pm_cov"),
        ({"pm": PM, "pm_cov": [[1.0, 2.0], [2.0, 1.0]]}, "pm_cov"),
        ({"pm": [1.0, np.nan], "pm_cov": PM_COV}, "pm"),
        ({"rv": 10.0, "rv_err": 1.0}, "need `pm`"),
        ({"pm": PM, "pm_cov": PM_COV, "rv": 10.0}, "go together"),
        ({"pm": PM, "pm_cov": PM_COV, "pm_outlier": 1.0}, "pm_outlier"),
        ({"pm": PM, "pm_cov": PM_COV, "pm_floor": -1.0}, "pm_floor"),
    ],
)
def test_motion_inputs_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _prior(np.array([1.0]), **kwargs)


def test_a_proper_motion_needs_both_labels_and_one_sightline():
    with pytest.raises(ValueError, match="both `feh` and `loga`"):
        logp_galactic_structure(
            np.array([1.0]), COORD, feh=np.array([0.0]), pm=PM, pm_cov=PM_COV
        )
    with pytest.raises(ValueError, match="one sightline"):
        logp_galactic_structure(
            np.array([1.0, 2.0]),
            (np.array([30.0, 40.0]), np.array([5.0, 5.0])),
            feh=np.zeros(2),
            loga=np.full(2, 9.5),
            pm=PM,
            pm_cov=PM_COV,
        )


def test_bulge_moments_follow_the_bv24_table_and_scale_with_the_potential():
    # At a table node, with R_solar = BV24's R0 and v_c = theirs, the moments
    # are the tabulated ones; other R_solar and v_c rescale radii and
    # velocities as the potential does.
    i, j = 10, 7
    R, Z = np.array([i * kin._BULGE_STEP]), np.array([j * kin._BULGE_STEP])
    mean, cov = bulge_velocity_moments(
        R, Z, R_solar=kin._BV24_R0, v_circ_solar=kin._V_VC0
    )
    np.testing.assert_allclose(
        [mean[0, 1], cov[0, 0, 0], cov[0, 1, 1], cov[0, 2, 2]],
        kin._BULGE[i, j],
        rtol=1e-12,
    )
    scaled = bulge_velocity_moments(
        2 * R, Z, R_solar=2 * kin._BV24_R0, v_circ_solar=2 * kin._V_VC0
    )
    np.testing.assert_allclose(scaled[0], 2 * mean, rtol=1e-12)
    np.testing.assert_allclose(scaled[1], 4 * cov, rtol=1e-12)


def test_bulge_moments_are_held_beyond_the_table():
    edge = (kin._BULGE.shape[0] - 1) * kin._BULGE_STEP
    near = bulge_velocity_moments(
        np.array([edge]), np.array([edge]), R_solar=kin._BV24_R0
    )
    far = bulge_velocity_moments(
        np.array([3 * edge]), np.array([2 * edge]), R_solar=kin._BV24_R0
    )
    np.testing.assert_array_equal(near[0], far[0])
    np.testing.assert_array_equal(near[1], far[1])
