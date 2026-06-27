#!/usr/bin/env python
"""
Direct measurement of MC-integration variance: plain vs antithetic sampling.

For a fixed object and fixed loglike result, the per-model integrated
log-posterior `lnp` returned by logpost_grid is a Monte-Carlo estimate. Running
logpost_grid M times with different seeds and measuring the seed-to-seed std of
`lnp` isolates the MC-integration noise (everything else is fixed). Antithetic
sampling should reduce this std without biasing the mean.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import harness as H  # noqa: E402

from brutus.utils import sampling  # noqa: E402


def measure(M=40, Nmc=50, nobj=6):
    s = H.load_setup()
    bf = s["bf"]
    phot_all, err_all = H._apply_offsets(s)
    good = np.where(s["mask"].all(axis=1) & np.isfinite(s["parallax"]))[0][:nobj]
    H._warmup(
        bf,
        phot_all[good[0]],
        err_all[good[0]],
        s["mask"][good[0]],
        s["parallax"][good[0]],
        s["parallax_err"][good[0]],
        tuple(s["coords"][good[0]]),
    )

    orig = sampling.sample_multivariate_normal

    def run_mode(antithetic):
        # Monkeypatch the binding used inside individual.py
        import brutus.analysis.individual as ind

        def patched(
            mean,
            cov,
            size=1,
            eps=1e-30,
            rstate=None,
            antithetic=None,
            _force=antithetic,
        ):
            # Ignore the caller's antithetic flag; force this mode.
            return orig(mean, cov, size=size, eps=eps, rstate=rstate, antithetic=_force)

        ind.sample_multivariate_normal = patched
        stds = []
        means_bias = []
        for i in good:
            i = int(i)
            lr = bf.loglike_grid(
                phot_all[i],
                err_all[i],
                s["mask"][i],
                return_vals=True,
                parallax=s["parallax"][i],
                parallax_err=s["parallax_err"][i],
            )
            lnps = []
            for m in range(M):
                r = bf.logpost_grid(
                    lr,
                    parallax=s["parallax"][i],
                    parallax_err=s["parallax_err"][i],
                    coord=tuple(s["coords"][i]),
                    Nmc_prior=Nmc,
                    wt_thresh=1e-3,
                    rstate=np.random.RandomState(m),
                )
                lnps.append(r[2])  # lnp per selected model
            lnps = np.array(lnps)  # (M, Nsel)
            stds.append(np.median(np.std(lnps, axis=0)))  # MC std of lnp
            means_bias.append(np.mean(lnps, axis=0))  # for bias check
        ind.sample_multivariate_normal = orig
        return np.array(stds), means_bias

    std_plain, mean_plain = run_mode(False)
    std_anti, mean_anti = run_mode(True)

    print(f"=== MC-integration std of lnp (median over models), M={M} repeats ===")
    print(
        f"{'obj':>4s} {'std_plain':>11s} {'std_anti':>11s} {'ratio':>7s} "
        f"{'max|mean shift|':>15s}"
    )
    for k in range(len(good)):
        bias = np.max(np.abs(mean_plain[k] - mean_anti[k]))
        ratio = std_anti[k] / max(std_plain[k], 1e-30)
        print(
            f"{int(good[k]):>4d} {std_plain[k]:11.4e} {std_anti[k]:11.4e} "
            f"{ratio:7.3f} {bias:15.4e}"
        )
    print(
        f"\nmean variance-reduction ratio (anti/plain): "
        f"{np.mean(std_anti/np.maximum(std_plain,1e-30)):.3f}  "
        f"(<1 = antithetic is less noisy)"
    )


if __name__ == "__main__":
    measure()
