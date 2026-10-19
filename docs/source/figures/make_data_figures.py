"""Draw the docs figures that come from model data, as themable SVG.

Run by hand, not by the docs build: ``python make_data_figures.py GRID OFFSETS``,
where GRID is ``grid_mist_v9.h5`` from ``brutus.data.fetch_grids`` and OFFSETS is
``offsets_mist_v9.txt`` from ``brutus.data.fetch_offsets``. It also fits the
Orion stars of the tutorials, which takes a few minutes. Each figure records the
git blob hashes of the figure scripts, the brutus version, the data files with
their SHA-256 and the seed in a comment. The figures use the helpers
of ``make_figures.py``, so they follow the light and dark themes.
"""

import hashlib
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from make_figures import axes, c, path, svg, text
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2

import brutus
from brutus.analysis import BruteForce
from brutus.core import StarGrid, get_seds
from brutus.data import load_models, load_offsets
from brutus.data.filters import gaia, ps, tmass
from brutus.utils import add_mag, inv_magnitude, magnitude

OUT = Path(__file__).parent
ORION = OUT.parents[2] / "tutorials" / "Orion_l209.1_b-19.9.h5"
SEED = 42
# MIST primary EEPs (Dotter 2016) that the default grid spans.
PRIMARY = {
    202: "ZAMS",
    353: "IAMS",
    454: "TAMS",
    605: "RGB tip",
    631: "ZACHeB",
    707: "TACHeB",
    808: "TP-AGB",
}


def solar_slice(grid_file):
    with h5py.File(grid_file, "r") as f:
        labels = f["labels"][:]
        params = f["parameters"][:]
        fehs = np.unique(labels["feh"])
        feh0 = fehs[np.argmin(np.abs(fehs))]
        sel = (labels["feh"] == feh0) & (labels["smf"] == 0)  # single stars only
        # Unreddened Gaia magnitudes at 1 kpc (m0 of mag_coeffs), shape (3, N).
        coeffs = f["mag_coeffs"].fields(gaia)[:]
        mags = np.array([coeffs[b][sel, 0] for b in gaia])
    return labels[sel], params[sel], mags, feh0


def isochrone(labels, params, mags, loga):
    """Interpolate the tracks to one age at each EEP: rows EEP, mass, G, BP, RP."""
    rows = []
    for eep in np.unique(labels["eep"]):
        on = labels["eep"] == eep
        age = params["loga"][on]
        order = np.argsort(age)
        age = age[order]
        if age[0] <= loga <= age[-1]:
            cols = np.vstack([labels["mini"][on], mags[:, on]])[:, order]
            rows.append([eep] + [np.interp(loga, age, x) for x in cols])
    return np.array(rows).T


def hr_tracks(labels, params, feh0):
    W, H = 760, 420
    x0, y0, x1, y1 = 80, 30, 600, 370
    tracks = [
        (0.6, "red"),
        (0.8, "copper"),
        (1.0, "amber"),
        (1.3, "amber"),
        (1.6, "copper"),
        (2.0, "red"),
    ]
    logt, logl = params["logt"], params["logl"]
    tmin, tmax, lmin, lmax = 3.45, 4.05, -1.4, 3.6

    def X(t):  # hotter to the left
        return x1 - (t - tmin) / (tmax - tmin) * (x1 - x0)

    def Y(lum):
        return y1 - (lum - lmin) / (lmax - lmin) * (y1 - y0)

    out = axes(
        x0, y0, x1, y1, "effective temperature (hotter ←)", "luminosity (log L/L☉)"
    )
    for m, col in tracks:
        on = np.isclose(labels["mini"], m)
        order = np.argsort(labels["eep"][on])
        t, lum, eep = logt[on][order], logl[on][order], labels["eep"][on][order]
        keep = (t > tmin) & (t < tmax) & (lum > lmin) & (lum < lmax)
        out += path(
            [(X(a), Y(b)) for a, b in zip(t[keep], lum[keep])], stroke=c(col), width=2
        )
        start = np.argmax(keep)
        out += text(
            X(t[start]) - 8, Y(lum[start]) + 4, f"{m} M☉", anchor="end", size=11
        )
        for e, name in PRIMARY.items():
            # Primary EEPs fall between grid points: interpolate along the track.
            te, le = np.interp(e, eep, t), np.interp(e, eep, lum)
            if e <= eep[-1] and tmin < te < tmax and lmin < le < lmax:
                out += f'<circle cx="{X(te):.1f}" cy="{Y(le):.1f}" r="4" fill="currentColor"/>'
                if np.isclose(m, 1.0):
                    out += text(X(te) + 8, Y(le) + 4, name, anchor="start", size=11)
    out += text(640, 60, f"[Fe/H] = {feh0:+.2f}", anchor="start", size=12)
    out += text(640, 84, "dots: primary EEPs,", anchor="start", size=12)
    out += text(640, 102, "labelled on 1 M☉", anchor="start", size=12)
    return svg(
        "hr-tracks",
        W,
        H,
        out,
        "MIST evolutionary tracks in the default grid",
        "Tracks of the default brutus grid at solar metallicity for initial masses from 0.6 to 2 solar masses, with the primary EEPs marked.",
    )


def grid_coverage(labels, feh0):
    W, H = 760, 380
    x0, y0, x1, y1 = 80, 30, 640, 320
    mini, eep = labels["mini"], labels["eep"]
    mlo, mhi, elo, ehi = mini.min(), mini.max(), eep.min(), eep.max()
    steps = np.unique(np.round(np.diff(np.unique(eep)), 6))[::-1]

    def X(m):
        return x0 + (m - mlo) / (mhi - mlo) * (x1 - x0)

    def Y(e):
        return y1 - (e - elo) / (ehi - elo) * (y1 - y0)

    out = axes(x0, y0, x1, y1, f"initial mass, {mlo:.1f} to {mhi:.1f} M☉", "EEP")
    for e, name in PRIMARY.items():
        out += path(
            [(x0, Y(e)), (x1, Y(e))], width=0.8, dash="3 4", extra='opacity="0.4"'
        )
        out += text(x1 + 8, Y(e) + 4, f"{e} {name}", anchor="start", size=11)
    for m in np.unique(mini):
        top = eep[mini == m].max()  # older models are cut at 13.8 Gyr
        out += path([(X(m), Y(elo)), (X(m), Y(top))], stroke=c("amber"), width=3)
    return svg(
        "grid-coverage",
        W,
        H,
        out,
        "Coverage of the default grid",
        f"Models of the default grid at [Fe/H] = {feh0:+.2f}: initial mass from {mlo:.1f} to {mhi:.1f} solar masses, "
        f"EEP from {elo:.0f} to {ehi:.0f} in steps of {' then '.join(f'{x:g}' for x in steps)}; "
        "models older than 13.8 Gyr are absent.",
    )


def cluster_cmd(labels, params, mags, feh0):
    W, H = 760, 420
    x0, y0, x1, y1 = 80, 30, 520, 370
    iso = isochrone(labels, params, mags, loga=9.6)  # 4 Gyr, 1 kpc
    eep, mass, bands = iso[0], iso[1], iso[2:]
    # Up to the tip of the red giant branch, within the plotted colours.
    rgb = (eep <= 605) & (bands[1] - bands[2] < 2.4)
    ms = eep <= 480  # eep_binary_max: companions only on the main sequence
    ms_mass = mass[ms]
    cmin, cmax, gmin, gmax = 0.4, 2.4, 8.0, 19.5

    def X(col):
        return x0 + (col - cmin) / (cmax - cmin) * (x1 - x0)

    def Y(mag):
        return y0 + (mag - gmin) / (gmax - gmin) * (y1 - y0)

    def on_ms(m):
        """Main-sequence magnitudes (G, BP, RP) of initial masses m."""
        return np.array([np.interp(m, ms_mass, b[ms]) for b in bands])

    def cmd_path(m, **kw):
        return path([(X(b - r), Y(g)) for g, b, r in m.T], **kw)

    def dot(x, y, filled):
        style = f"fill:{c('amber')}" if filled else f"fill:none;stroke:{c('red')}"
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" style="{style}" stroke-width="1.5"/>'

    rng = np.random.default_rng(SEED)
    n_members, n_field, noise = 120, 60, 0.03
    # Masses uniform in log mass; 30% of members have a companion with mass
    # ratio 0.6 to 1. Companions below the grid's lowest mass are left out.
    m1 = np.exp(rng.uniform(*np.log(ms_mass[[0, -1]]), n_members))
    q = rng.uniform(0.6, 1.0, n_members)
    binary = (rng.random(n_members) < 0.3) & (q * m1 >= ms_mass.min())
    members = on_ms(m1)
    # A companion adds its flux in each band, as StellarPop.get_seds does.
    members[:, binary] = add_mag(members[:, binary], on_ms(q * m1)[:, binary])
    members += rng.normal(0, noise, members.shape)
    field_colour = rng.uniform(cmin + 0.1, cmax - 0.1, n_field)
    field_g = rng.uniform(gmin + 1.5, gmax - 0.3, n_field)

    out = axes(x0, y0, x1, y1, "colour BP − RP (redder →)", "magnitude G (fainter ↓)")
    equal = add_mag(bands[:, ms], bands[:, ms])
    out += cmd_path(equal, stroke=c("copper"), width=2, dash="6 4")
    out += cmd_path(bands[:, rgb], width=2.5)
    out += "".join(dot(X(b - r), Y(g), True) for g, b, r in members.T)
    out += "".join(dot(X(col), Y(g), False) for col, g in zip(field_colour, field_g))

    lx, ly = 550, 60
    marks = [
        lambda y: path([(lx, y), (lx + 24, y)], width=2.5),
        lambda y: path(
            [(lx, y), (lx + 24, y)], stroke=c("copper"), width=2, dash="6 4"
        ),
        lambda y: dot(lx + 12, y, True),
        lambda y: dot(lx + 12, y, False),
    ]
    names = (
        "isochrone, single stars",
        "equal-mass binaries",
        "cluster members",
        "field stars",
    )
    for i, (mark, label) in enumerate(zip(marks, names)):
        y = ly + 24 * i
        out += mark(y) + text(lx + 34, y + 4, label, anchor="start", size=12)
    out += text(lx, ly + 120, f"4 Gyr, [Fe/H] = {feh0:+.2f},", anchor="start", size=12)
    out += text(lx, ly + 138, "1 kpc, no dust", anchor="start", size=12)
    return svg(
        "cluster-cmd",
        W,
        H,
        out,
        "A simulated star cluster",
        "Colour-magnitude diagram of a simulated 4 Gyr cluster at 1 kpc from the BruteForce model grid: "
        "the single-star isochrone, the equal-mass binary sequence 0.75 mag above it, members with and "
        "without companions, and field stars spread across the diagram.",
    )


def offsets_chart(offsets_file):
    names, values = np.loadtxt(offsets_file, dtype=str, unpack=True)
    percent = (values.astype(float) - 1) * 100
    surveys, bands = zip(*(n.split("_", 1) for n in names))
    group = {s: i for i, s in enumerate(dict.fromkeys(surveys))}
    W, H = 820, 330
    x0, y0, x1, y1 = 80, 30, 800, 220
    lim = 7.5
    step = (x1 - x0) / len(names)

    def Y(v):
        return (y0 + y1) / 2 - v / lim * (y1 - y0) / 2

    out = axes(x0, y0, x0, y1, "", "offset − 1 (%)", ylabel_gap=36)
    out += path([(x0, Y(0)), (x1, Y(0))], width=1)
    for v in (-6, -3, 3, 6):
        out += path(
            [(x0, Y(v)), (x1, Y(v))], width=0.6, dash="2 4", extra='opacity="0.5"'
        )
        out += text(x0 - 6, Y(v) + 4, f"{v:+.0f}", anchor="end", size=11)
    colours = (c("amber"), c("copper"))
    for i, (survey, band, v) in enumerate(zip(surveys, bands, percent)):
        x = x0 + (i + 0.5) * step
        top, bottom = sorted((Y(v), Y(0)))
        out += (
            f'<rect x="{x - step * 0.35:.1f}" y="{top:.1f}" width="{step * 0.7:.1f}" '
            f'height="{max(bottom - top, 1):.1f}" style="fill:{colours[group[survey] % 2]}"/>'
        )
        band = band.replace("_MAWf", "").replace("_MAW", "")
        out += (
            f'<text transform="translate({x + 4:.1f} {y1 + 8:.1f}) rotate(60)" '
            f'font-size="11" fill="currentColor">{survey} {band}</text>'
        )
    return svg(
        "offsets",
        W,
        H,
        out,
        "Default photometric offsets",
        "The multiplicative flux offsets of the default grid, as a percentage, for "
        f"{len(names)} bands from Gaia to WISE; they range from {percent.min():+.0f}% to {percent.max():+.0f}%.",
    )


def fit_orion(grid_file, offsets_file):
    """Fit the Orion stars of tutorial 5 with default settings, offset errors and seed."""
    filters = ps[:5] + tmass
    with h5py.File(ORION, "r") as f:
        stars = f["photometry"]["pixel 0-0"][:]
    mask = np.isfinite(stars["err"])
    # Missing bands hold mag = -999 and err = inf; give them harmless values.
    mag = np.where(mask, stars["mag"], 0.0)
    flux, err = inv_magnitude(mag, np.where(mask, stars["err"], 1.0))
    # Offset uncertainties, paper Table 5: 0.02 mag optical, 0.03 mag near-infrared.
    sys_mag = np.array([0.02] * (len(filters) - len(tmass)) + [0.03] * len(tmass))
    err = np.hypot(err, flux * sys_mag * np.log(10) / 2.5)
    parallax = stars["parallax"].astype(float)
    parallax_err = stars["parallax_error"].astype(float)
    bad = (parallax_err <= 0) | (parallax_err > 1e6) | (parallax == 0)
    parallax[bad] = parallax_err[bad] = np.nan

    models, labels, _ = load_models(grid_file, filters=filters, verbose=False)
    offsets = load_offsets(offsets_file, filters=filters, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "orion.h5"
        BruteForce(StarGrid(models, labels, verbose=False), verbose=False).fit(
            flux,
            err,
            mask,
            stars["obj_id"],
            str(out),
            phot_offsets=offsets,
            parallax=parallax,
            parallax_err=parallax_err,
            data_coords=np.c_[stars["l"], stars["b"]],
            rstate=np.random.RandomState(SEED),
            verbose=False,
        )
        with h5py.File(out, "r") as f:
            fit = {k: f[k][:] for k in f}
    fit["pvalue"] = chi2.sf(fit["obj_chi2min"], np.maximum(fit["obj_Nbands"] - 3, 1))
    # The data as the fit saw them: offsets applied to fluxes and errors.
    fit.update(
        flux=flux * offsets,
        err=err * offsets,
        mask=mask,
        models=models,
        parallax=parallax,
        parallax_err=parallax_err,
        filters=filters,
    )
    return fit


def orion_sightline(fit):
    W, H = 760, 380
    x0, y0, x1, y1 = 80, 30, 640, 320
    dmax, amax = 2.5, 6.0
    dist, av = fit["samps_dist"], fit["samps_red"]
    med_d, med_a = np.median(dist, axis=1), np.median(av, axis=1)
    good = (fit["pvalue"] > 1e-3) & (med_d < dmax)

    def X(d):
        return x0 + d / dmax * (x1 - x0)

    def Y(a):
        return y1 - a / amax * (y1 - y0)

    rng = np.random.default_rng(SEED)
    out = axes(x0, y0, x1, y1, "distance (kpc)", "extinction A(V) (mag)", 40, 36)
    for d in np.arange(0.5, dmax + 0.1, 0.5):
        out += text(X(d), y1 + 16, f"{d:g}", size=11)
    for a in range(0, int(amax) + 1, 2):
        out += text(x0 - 8, Y(a) + 4, f"{a}", anchor="end", size=11)
    for i in np.flatnonzero(good):
        for j in rng.choice(dist.shape[1], 25, replace=False):
            if dist[i, j] < dmax and av[i, j] < amax:
                out += (
                    f'<circle cx="{X(dist[i, j]):.1f}" cy="{Y(av[i, j]):.1f}" r="1.3" '
                    f'style="fill:{c("amber")}" opacity="0.35"/>'
                )
    for i in np.flatnonzero(good):
        out += (
            f'<circle cx="{X(med_d[i]):.1f}" cy="{Y(med_a[i]):.1f}" '
            f'r="3" style="fill:{c("red")}"/>'
        )
    lx = 655
    out += f'<circle cx="{lx + 6}" cy="60" r="3" style="fill:{c("red")}"/>'
    out += text(lx + 16, 64, "median", anchor="start", size=12)
    out += f'<circle cx="{lx + 6}" cy="84" r="2" style="fill:{c("amber")}"/>'
    out += text(lx + 16, 88, "draws", anchor="start", size=12)
    out += text(lx, 130, f"{good.sum()} stars,", anchor="start", size=12)
    out += text(lx, 148, "p > 0.001,", anchor="start", size=12)
    out += text(lx, 166, f"d < {dmax:g} kpc", anchor="start", size=12)
    return svg(
        "orion-sightline",
        W,
        H,
        out,
        "Distances and extinctions of stars towards Orion",
        f"Posterior draws and medians of distance and extinction for {good.sum()} well-fitted stars "
        f"within {dmax:g} kpc towards l = 209.1, b = -19.9. Extinction is low out to about 0.4 kpc "
        "and rises, with scatter, to 2 to 5 mag beyond 0.5 kpc, behind the Orion molecular cloud.",
    )


def star_posterior(fit):
    W, H = 820, 360
    snr = fit["parallax"] / fit["parallax_err"]
    dist, av = fit["samps_dist"], fit["samps_red"]
    # A well-fitted star behind the cloud with a moderate parallax.
    pick = (
        (fit["pvalue"] > 0.05) & (snr > 5) & (snr < 15) & (np.median(av, axis=1) > 1.5)
    )
    if not pick.any():
        raise ValueError(
            "no star meets the selection of star_posterior; relax its cuts"
        )
    i = np.flatnonzero(pick)[0]
    d, a = dist[i], av[i]

    # Left: the joint posterior of distance and A(V), with the parallax.
    x0, y0, x1, y1 = 70, 30, 360, 300
    dlo, dhi = np.quantile(d, [0.005, 0.995]) * [0.8, 1.2]
    alo, ahi = np.quantile(a, [0.005, 0.995]) + [-0.3, 0.3]

    def X(v):
        return x0 + (v - dlo) / (dhi - dlo) * (x1 - x0)

    def Y(v):
        return y1 - (v - alo) / (ahi - alo) * (y1 - y0)

    out = axes(x0, y0, x1, y1, "distance (kpc)", "A(V) (mag)", 40, 40)
    near, far = 1 / (fit["parallax"][i] + np.array([1, -1]) * fit["parallax_err"][i])
    left, right = X(max(near, dlo)), X(min(far, dhi))
    out += f'<rect x="{left:.1f}" y="{y0}" width="{right - left:.1f}" height="{y1 - y0}" style="fill:{c("copper")}" opacity="0.15"/>'
    for v in MaxNLocator(4).tick_values(dlo, dhi):
        if dlo < v < dhi:
            out += text(X(v), y1 + 16, f"{v:g}", size=11)
    for v in MaxNLocator(4).tick_values(alo, ahi):
        if alo < v < ahi:
            out += text(x0 - 8, Y(v) + 4, f"{v:g}", anchor="end", size=11)
    out += "".join(
        f'<circle cx="{X(dv):.1f}" cy="{Y(av_):.1f}" r="2" style="fill:{c("amber")}" opacity="0.6"/>'
        for dv, av_ in zip(d, a)
    )
    out += text((x0 + x1) / 2, 20, "posterior draws; band: parallax ± 1σ", size=12)

    # Right: residuals of the drawn models in each band, against the data errors.
    x0, y0, x1, y1 = 480, 30, 800, 300
    obs, obs_err = magnitude(fit["flux"][i], fit["err"][i])
    used = fit["mask"][i] & (obs_err < 0.25)  # the bands the fit kept (merr_max)
    mags = get_seds(fit["models"][fit["model_idx"][i]], av=a, rv=fit["samps_dred"][i])
    mags += 5 * np.log10(d)[:, None]
    lo, hi = np.quantile(mags - obs, [0.025, 0.975], axis=0)
    lim = 0.3
    step = (x1 - x0) / len(used)

    def Yr(r):
        return (y0 + y1) / 2 - r / lim * (y1 - y0) / 2

    out += axes(x0, y0, x0, y1, "", "model − data (mag)", ylabel_gap=40)
    out += path([(x0, Yr(0)), (x1, Yr(0))], width=1)
    for r in (-0.2, -0.1, 0.1, 0.2):
        out += text(x0 - 8, Yr(r) + 4, f"{r:+.1f}", anchor="end", size=11)
    for b, name in enumerate(fit["filters"]):
        if not used[b]:
            continue
        x = x0 + (b + 0.5) * step
        top, bottom = Yr(obs_err[b]), Yr(-obs_err[b])
        out += f'<rect x="{x - 12:.1f}" y="{top:.1f}" width="24" height="{bottom - top:.1f}" style="fill:{c("copper")}" opacity="0.2"/>'
        out += path([(x, Yr(lo[b])), (x, Yr(hi[b]))], stroke=c("amber"), width=6)
        out += text(x, y1 + 16, name.replace("2MASS_", "").replace("PS_", ""), size=11)
    out += text((x0 + x1) / 2, 20, "bars: 95% of model draws; band: data ± 1σ", size=12)
    return svg(
        "star-posterior",
        W,
        H,
        out,
        "The posterior of one star",
        f"One star towards Orion, fitted with a parallax of {fit['parallax'][i]:.2f} ± {fit['parallax_err'][i]:.2f} mas. "
        "Left: posterior draws of distance and extinction, against the parallax range. "
        "Right: in each band, the range of the drawn models' magnitudes minus the observed magnitude, against the data error.",
    )


@lru_cache
def sha256(file):
    """First 10 hex digits of a file's SHA-256, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:10]


def save(name, figure, sources, made_by):
    files = ", ".join(f"{Path(f).name} {sha256(f)}" for f in sources)
    note = f"<!-- {made_by}; data: {files}; seed {SEED} -->\n"
    (OUT / f"{name}.svg").write_text(note + figure)


if __name__ == "__main__":
    grid_file, offsets_file = sys.argv[1:3]
    # Git blob hashes of both figure scripts identify the code and survive rebases.
    scripts = [__file__, OUT / "make_figures.py"]
    blobs = subprocess.run(
        ["git", "hash-object", *scripts], capture_output=True, text=True, check=True
    ).stdout.split()
    made_by = ", ".join(f"{Path(s).name} {b[:10]}" for s, b in zip(scripts, blobs))
    made_by += f", brutus {brutus.__version__}"
    labels, params, mags, feh0 = solar_slice(grid_file)
    save("hr-tracks", hr_tracks(labels, params, feh0), [grid_file], made_by)
    save("grid-coverage", grid_coverage(labels, feh0), [grid_file], made_by)
    save("cluster-cmd", cluster_cmd(labels, params, mags, feh0), [grid_file], made_by)
    save("offsets", offsets_chart(offsets_file), [offsets_file], made_by)
    fit = fit_orion(grid_file, offsets_file)
    sources = [grid_file, offsets_file, ORION]
    save("orion-sightline", orion_sightline(fit), sources, made_by)
    save("star-posterior", star_posterior(fit), sources, made_by)
