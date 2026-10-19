"""Draw the docs figures that come from model data, as themable SVG.

Run by hand, not by the docs build: ``python make_data_figures.py GRID OFFSETS``,
where GRID is ``grid_mist_v9.h5`` from ``brutus.data.fetch_grids`` and OFFSETS is
``offsets_mist_v9.txt`` from ``brutus.data.fetch_offsets``. The figures use
the helpers of ``make_figures.py``, so they follow the light and dark themes.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from make_figures import axes, c, path, svg, text

from brutus.data.filters import gaia
from brutus.utils import add_mag

OUT = Path(__file__).parent
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

    rng = np.random.default_rng(3)
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


if __name__ == "__main__":
    labels, params, mags, feh0 = solar_slice(sys.argv[1])
    (OUT / "hr-tracks.svg").write_text(hr_tracks(labels, params, feh0))
    (OUT / "grid-coverage.svg").write_text(grid_coverage(labels, feh0))
    (OUT / "cluster-cmd.svg").write_text(cluster_cmd(labels, params, mags, feh0))
    (OUT / "offsets.svg").write_text(offsets_chart(sys.argv[2]))
