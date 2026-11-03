# brutus

:::::{div} hero

```{image} _static/brutus-logo-light.svg
:alt: brutus
:class: only-light
```

```{image} _static/brutus-logo-dark.svg
:alt: brutus
:class: only-dark
```

:::{div} hero-tagline
`brutus` infers Bayesian posteriors for stellar distances, dust extinctions, and intrinsic parameters from broadband photometry and astrometry by evaluating and marginalizing over pre-computed stellar model grids.
:::

::::{div} hero-buttons

```{button-ref} quickstart
:color: primary
Get started
```

```{button-ref} tutorials/index
:color: primary
:outline:
Tutorials
```

```{button-ref} api/index
:color: primary
:outline:
API reference
```

::::

:::::

## Core Capabilities

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Fit individual stars
:link: quickstart
:link-type: doc

Infer joint posteriors over distance, extinction ($A_V$, $R_V$), and intrinsic stellar parameters across large photometric and astrometric catalogs using 3D Galactic structure and dust-map priors.
:::

:::{grid-item-card} Model star clusters
:link: population_modeling
:link-type: doc

Evaluate marginal likelihoods for coeval stellar populations with shared age, metallicity, distance, and foreground extinction while accounting for unresolved binaries and field-star contamination.
:::

:::{grid-item-card} Map dust in 3D
:link: tutorials/index
:link-type: doc

Combine stellar distance-extinction posteriors along a sightline to infer the distances, amplitudes, and widths of foreground interstellar dust clouds.
:::

::::

## Inference Pipeline

:::{div} steps
1. **Evaluate grid likelihoods** across every template in a pre-computed stellar evolution grid (`StarGrid`).
2. **Optimize extrinsic parameters** by solving for the maximum-likelihood distance scale, visual extinction $A_V$, and reddening vector shape $R_V$ (and their local $3 \times 3$ Fisher information matrix) at each grid node.
3. **Integrate astrophysical priors** over distance, 3D dust extinction, parallax, the initial mass function (IMF), and evolutionary lifetimes via antithetic Monte Carlo importance sampling.
4. **Resample the joint posterior** across grid models and continuous extrinsic parameters without requiring iterative Markov chain Monte Carlo (MCMC) burn-in or convergence diagnostics ({doc}`scientific_background`).
:::

## Minimal Example

```python
from brutus.analysis import BruteForce
from brutus.core import StarGrid
from brutus.data import fetch_grids, load_models
from brutus.utils import inv_magnitude

models, labels, _ = load_models(fetch_grids(grid="mist_v9"), filters=filters)
fitter = BruteForce(StarGrid(models, labels))

flux, flux_err = inv_magnitude(mag, mag_err)  # linear flux densities in maggies
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    parallax=parallax, parallax_err=parallax_err,  # in mas
    data_coords=coords,  # Galactic (l, b) in degrees
    save_file="results.h5",
)
```

See {doc}`quickstart` for a complete end-to-end workflow including systematic error floors and posterior diagnostics.

## Citing brutus

If you use `brutus` in your research, please cite [Speagle et al. (2025), arXiv:2503.02227](https://arxiv.org/abs/2503.02227).

```{toctree}
:caption: Getting started
:hidden:

installation
quickstart
glossary
```

```{toctree}
:caption: Scientific background
:hidden:

scientific_background
stellar_models
priors
grid_generation
population_modeling
photometric_offsets
```

```{toctree}
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:caption: User guide
:hidden:

understanding_results
faq
```

```{toctree}
:caption: Reference
:hidden:

api/index
references
migration
changelog
contributing
```
