#!/usr/bin/env python
"""
Accuracy vs a high-Nmc gold standard: is antithetic MORE correct than plain?

The per-model integrated log-posterior `lnp` at finite Nmc is a biased, noisy
estimate of the true (Nmc->inf) value. We build a gold standard with very large
Nmc averaged over several seeds, then measure the RMSE of plain@Nmc and
antithetic@Nmc against it. Lower RMSE = more accurate.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import harness as H  # noqa: E402

from brutus.utils import sampling  # noqa: E402


def lnp_for(bf, lr, s, i, Nmc, seed, force):
    import brutus.analysis.individual as ind

    orig = sampling.sample_multivariate_normal

    def patched(mean, cov, size=1, eps=1e-30, rstate=None, antithetic=None, _f=force):
        return orig(mean, cov, size=size, eps=eps, rstate=rstate, antithetic=_f)

    ind.sample_multivariate_normal = patched
    r = bf.logpost_grid(
        lr,
        parallax=s["parallax"][i],
        parallax_err=s["parallax_err"][i],
        coord=tuple(s["coords"][i]),
        Nmc_prior=Nmc,
        wt_thresh=1e-3,
        rstate=np.random.RandomState(seed),
    )
    ind.sample_multivariate_normal = orig
    return r[0], r[2]  # sel, lnp


def main(nobj=6, Nmc=50, Ngold=4000, ngoldseeds=6, ntrial=12, objs=None):
    s = H.load_setup()
    bf = s["bf"]
    phot_all, err_all = H._apply_offsets(s)
    if objs is not None:
        good = np.array([int(i) for i in objs])
    else:
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
    print(
        f"=== accuracy vs gold (Nmc_gold={Ngold}x{ngoldseeds} seeds), "
        f"test Nmc={Nmc}, {ntrial} trials ==="
    )
    print(
        f"{'obj':>4s} {'RMSE_plain':>11s} {'RMSE_anti':>11s} {'ratio':>7s} "
        f"{'bias_plain':>11s} {'bias_anti':>11s}"
    )
    rp_all, ra_all = [], []
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
        # Gold: large Nmc, averaged over seeds (antithetic, lowest variance)
        sel0, _ = lnp_for(bf, lr, s, i, Nmc, 0, True)
        gold = np.zeros(len(sel0))
        for g in range(ngoldseeds):
            _, lg = lnp_for(bf, lr, s, i, Ngold, 1000 + g, True)
            gold += lg
        gold /= ngoldseeds
        # Trials at test Nmc
        ep, ea = [], []
        for t in range(ntrial):
            _, lp = lnp_for(bf, lr, s, i, Nmc, t, False)
            _, la = lnp_for(bf, lr, s, i, Nmc, t, True)
            ep.append(lp - gold)
            ea.append(la - gold)
        ep, ea = np.array(ep), np.array(ea)
        rmse_p = np.sqrt(np.mean(ep**2))
        rmse_a = np.sqrt(np.mean(ea**2))
        bias_p = np.mean(ep)  # mean error (Jensen bias)
        bias_a = np.mean(ea)
        rp_all.append(rmse_p)
        ra_all.append(rmse_a)
        print(
            f"{i:>4d} {rmse_p:11.4e} {rmse_a:11.4e} {rmse_a/rmse_p:7.3f} "
            f"{bias_p:11.3e} {bias_a:11.3e}"
        )
    rp_all, ra_all = np.array(rp_all), np.array(ra_all)
    print(
        f"\nmean RMSE ratio (anti/plain): {np.mean(ra_all/rp_all):.3f}  "
        f"(<1 => antithetic is MORE accurate vs the gold standard)"
    )


if __name__ == "__main__":
    main()
