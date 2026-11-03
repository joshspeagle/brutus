# Tutorials

The `tutorials/` directory contains 12 interactive Jupyter notebooks that walk through `brutus` from initial data setup and forward modeling to individual-star inference, cluster fitting, 3D dust mapping, and empirical photometric calibration. Because most notebooks use the MIST stellar grids, isochrones, and Orion tutorial datasets, run {doc}`tutorial_00_data_setup` first to verify your installation and download the required files.

## 1. Environment and Data Setup

| Notebook | Topics Covered |
|---|---|
| {doc}`tutorial_00_data_setup` | Verify the `brutus` installation, inspect the 49 supported photometric passbands, and download the MIST, Bayestar, and tutorial datasets |

```{toctree}
:hidden:

tutorial_00_data_setup
```

## 2. Forward Models and Astrophysical Priors

| Notebook | Topics Covered |
|---|---|
| {doc}`tutorial_01_individual_stars` | Predict physical stellar parameters and reddened broadband SEDs along MIST evolutionary tracks (`EEPTracks`, `StarEvolTrack`), including unresolved binaries |
| {doc}`tutorial_02_populations` | Interpolate MIST isochrones (`Isochrone`) and synthesize single and binary stellar population photometry (`StellarPop`) |
| {doc}`tutorial_03_grids_performance` | Generate, load, and inspect pre-computed HDF5 stellar grids (`GridGenerator`, `StarGrid`) and their linear reddening coefficients |
| {doc}`tutorial_04_galactic_priors` | Evaluate and visualize the Kroupa IMF, 3D Galactic structure, Bayestar dust-map, and astrometric parallax priors |

```{toctree}
:hidden:

tutorial_01_individual_stars
tutorial_02_populations
tutorial_03_grids_performance
tutorial_04_galactic_priors
```

## 3. Bayesian Inference Workflows

| Notebook | Topics Covered |
|---|---|
| {doc}`tutorial_05_fitting_individual` | Fit individual field stars in Orion with `BruteForce` under sequential prior configurations |
| {doc}`tutorial_06_cluster_analysis` | Infer the age, metallicity, distance, extinction, and field contamination of open cluster M67 using `isochrone_population_loglike` |
| {doc}`tutorial_07_dust_mapping` | Fit parametric 3D interstellar dust clouds along the Orion sightline from individual stellar $(d, A_V)$ posteriors |
| {doc}`tutorial_08_photometric_calibration` | Derive leave-one-out empirical photometric zero-point offsets with `photometric_offsets` |

```{toctree}
:hidden:

tutorial_05_fitting_individual
tutorial_06_cluster_analysis
tutorial_07_dust_mapping
tutorial_08_photometric_calibration
```

## 4. Utilities, Visualization, and Diagnostics

| Notebook | Topics Covered |
|---|---|
| {doc}`tutorial_09_utilities` | Convert magnitudes and fluxes, evaluate photometric likelihoods, and draw weighted posterior samples with `brutus.utils` |
| {doc}`tutorial_10_plotting` | Generate corner plots, 2D distance-reddening densities, and posterior predictive SED checks with `brutus.plotting` |
| {doc}`tutorial_11_results` | Inspect `BruteForce` HDF5 output files, compute credible intervals, and run $\chi^2$ and `mc_ess` fit diagnostics |

```{toctree}
:hidden:

tutorial_09_utilities
tutorial_10_plotting
tutorial_11_results
```
