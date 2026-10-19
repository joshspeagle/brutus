"""Draw the method schematics in docs/source/figures/.

Run by hand, not by the docs build: ``python make_figures.py``. The SVGs draw in
``currentColor`` and the brutus CSS variables, so they follow the light and
dark themes when included inline.
"""

import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent


def c(var):
    """Fill or stroke colour from a brutus CSS variable."""
    return f"var(--brutus-{var})"


def svg(name, w, h, body, title, desc):
    """Wrap a figure; ids carry the figure name so inline SVGs on one page differ."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
        f'aria-labelledby="{name}-t {name}-d">'
        f'<title id="{name}-t">{title}</title><desc id="{name}-d">{desc}</desc>'
        f"{arrowhead(name)}{body}</svg>\n"
    )


def arrowhead(name):
    return (
        f'<defs><marker id="{name}-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/>'
        "</marker></defs>"
    )


def text(x, y, s, anchor="middle", size=13, weight="normal", style=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="currentColor" style="{style}">{s}</text>'
    )


def path(pts, stroke="currentColor", width=2, fill="none", dash=None, extra=""):
    d = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" style="stroke:{stroke};fill:{fill}" stroke-width="{width}"'
        f' stroke-linejoin="round" stroke-linecap="round"{da} {extra}/>'
    )


def axes(x0, y0, x1, y1, xlabel, ylabel):
    out = path([(x0, y0), (x0, y1), (x1, y1)], width=1.5)
    out += text((x0 + x1) / 2, y1 + 22, xlabel, size=12)
    out += (
        f'<text transform="translate({x0 - 14:.1f} {(y0 + y1) / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-size="12" fill="currentColor">{ylabel}</text>'
    )
    return out


def ellipse_pts(cx, cy, a, b, ang, n=90):
    t = np.linspace(0, 2 * np.pi, n)
    ca, sa = math.cos(ang), math.sin(ang)
    return [
        (
            cx + a * math.cos(u) * ca - b * math.sin(u) * sa,
            cy + a * math.cos(u) * sa + b * math.sin(u) * ca,
        )
        for u in t
    ]


# -- 1. Forward model -----------------------------------------------------------------


def forward_model():
    W, H = 760, 330
    x0, y0, x1, y1 = 120, 30, 600, 270
    bands = [
        ("g", 0.48),
        ("r", 0.62),
        ("i", 0.75),
        ("z", 0.87),
        ("y", 0.96),
        ("J", 1.24),
        ("H", 1.66),
        ("Ks", 2.16),
    ]
    lam = np.array([b[1] for b in bands])
    T = 5000.0
    bb = 1 / lam**5 / (np.exp(14388 / (lam * T)) - 1)
    m_int = -2.5 * np.log10(bb / bb.max())  # model at 1 kpc
    m_dist = m_int + 5 * np.log10(2.0)  # moved to 2 kpc
    alam = (lam / 0.55) ** -1.6  # A_lambda / A_V
    m_obs = m_dist + 1.5 * alam  # A_V = 1.5
    lo, hi = m_int.min() - 0.3, m_obs.max() + 0.5

    def X(wl):
        return x0 + math.log(wl / 0.40) / math.log(2.3 / 0.40) * (x1 - x0)

    def Y(m):
        return y0 + (m - lo) / (hi - lo) * (y1 - y0)

    out = axes(x0, y0, x1, y1, "wavelength", "brightness")
    for name, wl in bands:
        out += text(X(wl), y1 + 38, name, size=11, style="opacity:0.7")
    out += path([(X(wl), Y(m)) for wl, m in zip(lam, m_int)], dash="5 5", width=2)
    out += path(
        [(X(wl), Y(m)) for wl, m in zip(lam, m_dist)], stroke=c("amber"), width=2.5
    )
    out += path(
        [(X(wl), Y(m)) for wl, m in zip(lam, m_obs)], stroke=c("red"), width=2.5
    )
    rng = np.random.default_rng(3)
    for wl, m in zip(lam, m_obs):
        mo = m + rng.normal(0, 0.06)
        out += path([(X(wl), Y(mo - 0.15)), (X(wl), Y(mo + 0.15))], width=1.5)
        out += f'<circle cx="{X(wl):.1f}" cy="{Y(mo):.1f}" r="4" fill="currentColor"/>'
    # Curve labels at the red end, kept apart vertically
    ys = [Y(m_int[-1]), Y(m_dist[-1]), Y(m_obs[-1])]
    for i in (1, 2):
        ys[i] = max(ys[i], ys[i - 1] + 18)
    for yl, s_ in zip(
        ys, ["model SED at 1 kpc", "moved to distance d", "dimmed by dust; data"]
    ):
        out += text(X(2.16) + 14, yl + 4, s_, anchor="start", size=12)
    # Arrows between the curves, labelled to their right
    la = 0.54
    mi, md, mo = (
        np.interp(math.log(la), np.log(lam), v) for v in (m_int, m_dist, m_obs)
    )
    xa = X(la)
    out += path(
        [(xa, Y(mi) + 6), (xa, Y(md) - 5)],
        width=1.5,
        extra='marker-end="url(#forward-model-ah)"',
    )
    out += path(
        [(xa, Y(md) + 6), (xa, Y(mo) - 5)],
        stroke=c("red"),
        width=1.5,
        extra='marker-end="url(#forward-model-ah)"',
    )
    out += text(
        xa + 8, (Y(mi) + Y(md)) / 2 + 4, "scale = (1 kpc / d)²", anchor="start", size=12
    )
    out += text(
        xa + 8, (Y(md) + Y(mo)) / 2 - 6, "dust: most in blue", anchor="start", size=12
    )
    return svg(
        "forward-model",
        W,
        H,
        out,
        "The forward model",
        "A model SED at 1 kpc is scaled to the distance of the star and dimmed by dust, which dims blue bands more than red ones, and compared with the observed photometry.",
    )


# -- 2. Pipeline -----------------------------------------------------------------------


def pipeline():
    W, H = 900, 300
    pw, gap, top = 195, 25, 60
    out = ""
    titles = [
        "1  Fit every model",
        "2  Keep the good fits",
        "3  Integrate the priors",
        "4  Draw the posterior",
    ]
    rng = np.random.default_rng(1)
    gx, gy = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 9))
    gx, gy = gx.ravel(), gy.ravel()
    chi = ((gx - 0.62) / 0.16) ** 2 + ((gy - 0.42 - 0.5 * (gx - 0.62)) / 0.12) ** 2
    w = np.exp(-0.5 * chi)
    for k in range(4):
        X0 = 15 + k * (pw + gap)
        out += f'<rect x="{X0}" y="{top - 10}" width="{pw}" height="{H - top - 10}" rx="10" style="fill:var(--color-background-secondary)"/>'
        out += text(X0 + pw / 2, top - 22, titles[k], size=14, weight="600")
        px, py = X0 + 22, top + 12
        sw, sh = pw - 44, 170
        if k in (0, 1):
            for xi, yi, wi in zip(gx, gy, w):
                cx, cy = px + xi * sw, py + (1 - yi) * sh
                if k == 0:
                    op = 0.25 + 0.75 * wi
                    out += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{3 + 4 * wi:.1f}" style="fill:{c("amber")}" opacity="{op:.2f}"/>'
                else:
                    keep = wi > 1e-3
                    col = c("red") if keep else "currentColor"
                    op = 1 if keep else 0.15
                    out += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{3 + 4 * wi if keep else 2.5:.1f}" style="fill:{col}" opacity="{op}"/>'
            out += text(
                px + sw / 2,
                py + sh + 30,
                "mass, age, metallicity" if k == 0 else "within 10⁻³ of the best",
                size=11,
                style="opacity:0.8",
            )
            if k == 0:
                out += text(
                    px + sw / 2,
                    py + sh + 46,
                    "best distance and dust each",
                    size=11,
                    style="opacity:0.8",
                )
        elif k == 2:
            # Prior weight in log distance (background bands): the volume and the
            # Jacobian make it rise with distance, the disk falloff makes it fall.
            def prior(u):
                return (u / 0.6) ** 3 * math.exp(3 * (1 - u / 0.6))

            for j in range(8):
                op = 0.04 + 0.32 * prior((j + 0.5) / 8)
                out += f'<rect x="{px + j * sw / 8:.1f}" y="{py}" width="{sw / 8 + 0.5:.1f}" height="{sh}" style="fill:{c("gold")}" opacity="{op:.2f}"/>'
            cx, cy, a, b, ang = px + sw * 0.55, py + sh * 0.5, 62, 20, math.radians(32)
            out += path(ellipse_pts(cx, cy, a, b, ang), stroke=c("red"), width=2)
            z = rng.normal(size=(12, 2))
            z = np.vstack([z, -z])
            for zx, zy in z:
                ex = cx + 0.9 * (zx * a * math.cos(ang) - zy * b * math.sin(ang))
                ey = cy + 0.9 * (zx * a * math.sin(ang) + zy * b * math.cos(ang))
                wt = prior(min(max((ex - px) / sw, 0.0), 1.0))
                out += f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{1.5 + 3.5 * wt:.1f}" fill="currentColor" opacity="0.85"/>'
            out += path([(px, py + sh), (px + sw, py + sh)], width=1)
            out += text(px + sw / 2, py + sh + 16, "log distance →", size=11)
            out += text(
                px + sw / 2,
                py + sh + 34,
                "draws in pairs (z, −z),",
                size=11,
                style="opacity:0.8",
            )
            out += text(
                px + sw / 2,
                py + sh + 50,
                "weighted by the priors",
                size=11,
                style="opacity:0.8",
            )
        else:
            xs = np.linspace(0, 1, 200)
            mus = rng.normal(0.5, 0.12, 9)
            sds = rng.uniform(0.04, 0.08, 9)
            wts = rng.dirichlet(np.ones(9) * 2)
            tot = sum(
                wt * np.exp(-0.5 * ((xs - m) / s) ** 2) / s
                for m, s, wt in zip(mus, sds, wts)
            )
            scale = sh * 0.85 / tot.max()
            for m, s, wt in zip(mus, sds, wts):
                comp = wt * np.exp(-0.5 * ((xs - m) / s) ** 2) / s
                out += path(
                    [(px + x * sw, py + sh - v * scale) for x, v in zip(xs, comp)],
                    stroke=c("amber"),
                    width=1.2,
                )
            out += path(
                [(px + x * sw, py + sh - v * scale) for x, v in zip(xs, tot)],
                stroke=c("red"),
                width=2.5,
            )
            out += path([(px, py + sh), (px + sw, py + sh)], width=1)
            out += text(px + sw / 2, py + sh + 16, "distance →", size=11)
            out += text(
                px + sw / 2,
                py + sh + 34,
                "the sum over models;",
                size=11,
                style="opacity:0.8",
            )
            out += text(
                px + sw / 2,
                py + sh + 50,
                "250 draws are saved",
                size=11,
                style="opacity:0.8",
            )
        if k < 3:
            ax = X0 + pw + 4
            out += path(
                [(ax, H / 2 + 10), (ax + gap - 8, H / 2 + 10)],
                width=2,
                extra='marker-end="url(#pipeline-ah)"',
            )
    return svg(
        "pipeline",
        W,
        H,
        out,
        "The four steps of a brutus fit",
        "Every grid model is fitted for distance and dust; the good fits are kept; for each, Monte Carlo draws in log-distance and dust are weighted by the priors; the posterior is the weighted sum over models.",
    )


# -- 3. Degeneracy ---------------------------------------------------------------------


def degeneracy():
    W, H = 820, 330
    x0, y0, x1, y1 = 80, 30, 520, 270
    out = axes(x0, y0, x1, y1, "distance", "extinction A(V)")
    # Photometry: a hotter star needs more dust and lies farther away, so
    # distance and extinction correlate positively.
    cx, cy = 300, 150
    out += path(
        ellipse_pts(cx, cy, 170, 22, math.radians(-28)),
        stroke=c("amber"),
        width=2.5,
        fill=c("amber"),
        extra='fill-opacity="0.18"',
    )
    # Parallax: a vertical band in distance
    out += f'<rect x="{cx - 70}" y="{y0}" width="58" height="{y1 - y0}" style="fill:{c("copper")}" opacity="0.15"/>'
    out += path(
        [(cx - 70, y0), (cx - 70, y1)], stroke=c("copper"), width=1.5, dash="4 4"
    )
    out += path(
        [(cx - 12, y0), (cx - 12, y1)], stroke=c("copper"), width=1.5, dash="4 4"
    )
    # Product
    out += path(
        ellipse_pts(cx - 41, cy + 22, 28, 13, math.radians(-28)),
        stroke=c("red"),
        width=2.5,
        fill=c("red"),
        extra='fill-opacity="0.35"',
    )
    # Legend
    lx, ly = 550, 70
    items = [
        (
            c("amber"),
            "photometry alone: nearer and",
            "less reddened, or farther and more",
        ),
        (c("copper"), "parallax: a range", "of distance"),
        (c("red"), "both: the posterior", ""),
    ]
    for i, (col, a, b) in enumerate(items):
        y = ly + i * 62
        out += f'<rect x="{lx}" y="{y - 11}" width="16" height="16" rx="3" style="fill:{col}" opacity="0.7"/>'
        out += text(lx + 26, y + 2, a, anchor="start", size=12)
        if b:
            out += text(lx + 26, y + 19, b, anchor="start", size=12)
    return svg(
        "degeneracy",
        W,
        H,
        out,
        "The distance-extinction degeneracy",
        "Photometry alone allows a nearer, less reddened star or a farther, more reddened one. A parallax limits the distance, and the posterior is where both agree.",
    )


for name, fn in (
    ("forward-model", forward_model),
    ("pipeline", pipeline),
    ("degeneracy", degeneracy),
):
    (OUT / f"{name}.svg").write_text(fn())
    print(name)
