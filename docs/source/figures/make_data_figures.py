"""Draw the docs figures that come from model data, as themable SVG.

Run by hand, not by the docs build: ``python make_data_figures.py GRID``, where
GRID is ``grid_mist_v9.h5`` from ``brutus.data.fetch_grids``. The figures use
the helpers of ``make_figures.py``, so they follow the light and dark themes.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from make_figures import axes, c, path, svg, text

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
    return labels[sel], params[sel], feh0


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


if __name__ == "__main__":
    labels, params, feh0 = solar_slice(sys.argv[1])
    (OUT / "hr-tracks.svg").write_text(hr_tracks(labels, params, feh0))
    (OUT / "grid-coverage.svg").write_text(grid_coverage(labels, feh0))
