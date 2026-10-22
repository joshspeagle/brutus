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
Distances, extinctions and stellar properties from photometry and astrometry, by evaluating every model on a grid.
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

## What brutus does

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Fit individual stars
:link: quickstart
:link-type: doc

Fit the distance, extinction and stellar properties of thousands of stars from their photometry and parallaxes, with Galactic and dust-map priors.
:::

:::{grid-item-card} Model star clusters
:link: population_modeling
:link-type: doc

Fit one age, metallicity, distance and extinction to a coeval population, with binaries and outliers.
:::

:::{grid-item-card} Map dust in 3D
:link: tutorials/index
:link-type: doc

Combine the distance-extinction posteriors of many stars to measure the dust along a line of sight.
:::

::::

## How it works

:::{div} steps
1. Evaluate the likelihood of the data for **every model** in a grid of stellar models.
2. For each model, fit the **distance and extinction** that match the data best.
3. Weight each model by its **priors**: Galactic structure, dust maps, parallax and initial mass function.
4. The posterior is the **weighted mixture** of all models. No sampler, no convergence checks.
:::

{doc}`scientific_background` explains each step.

## A first fit

```python
from brutus.analysis import BruteForce
from brutus.core import StarGrid
from brutus.data import fetch_grids, load_models
from brutus.utils import inv_magnitude

models, labels, _ = load_models(fetch_grids(grid="mist_v9"), filters=filters)
fitter = BruteForce(StarGrid(models, labels))

flux, flux_err = inv_magnitude(mag, mag_err)  # brutus fits fluxes, not magnitudes
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    parallax=parallax, parallax_err=parallax_err,  # in mas
    data_coords=coords,  # Galactic (l, b) in degrees
    save_file="results.h5",
)
```

The {doc}`quickstart` runs this end to end and reads the results.

## Citing brutus

If you use brutus, please cite [Speagle et al. (2025), arXiv:2503.02227](https://arxiv.org/abs/2503.02227).

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
