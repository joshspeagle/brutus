# Tutorials

The tutorials are Jupyter notebooks in the `tutorials/` folder of the repository. Most need downloaded model data: start with Tutorial 0.

## Setup

| Notebook | What it teaches |
|---|---|
| {doc}`tutorial_00_data_setup` | check the installation, the filter lists, and download the data files |

```{toctree}
:hidden:

tutorial_00_data_setup
```

## Models and priors

| Notebook | What it teaches |
|---|---|
| {doc}`tutorial_01_individual_stars` | predict stellar parameters and SEDs along evolutionary tracks, with binaries and extinction |
| {doc}`tutorial_02_populations` | build isochrones and synthetic star clusters |
| {doc}`tutorial_03_grids_performance` | generate, load and inspect model grids and their reddening coefficients |
| {doc}`tutorial_04_galactic_priors` | evaluate and combine the Galactic, dust and parallax priors |

```{toctree}
:hidden:

tutorial_01_individual_stars
tutorial_02_populations
tutorial_03_grids_performance
tutorial_04_galactic_priors
```

## Fitting

| Notebook | What it teaches |
|---|---|
| {doc}`tutorial_05_fitting_individual` | fit individual stars with `BruteForce` |
| {doc}`tutorial_06_cluster_analysis` | fit the cluster M67 with an isochrone, binaries and MCMC |
| {doc}`tutorial_07_dust_mapping` | fit dust clouds along a line of sight to stellar posteriors |
| {doc}`tutorial_08_photometric_calibration` | compute and apply photometric offsets |

```{toctree}
:hidden:

tutorial_05_fitting_individual
tutorial_06_cluster_analysis
tutorial_07_dust_mapping
tutorial_08_photometric_calibration
```

## Utilities and results

| Notebook | What it teaches |
|---|---|
| {doc}`tutorial_09_utilities` | convert photometry, evaluate likelihoods and sample with `brutus.utils` |
| {doc}`tutorial_10_plotting` | draw the figures of `brutus.plotting` |
| {doc}`tutorial_11_results` | read, check and summarize the output file of a fit |

```{toctree}
:hidden:

tutorial_09_utilities
tutorial_10_plotting
tutorial_11_results
```
