Installation
============

Requirements
------------

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL (see Windows note below)

Quick Install
-------------

For most users, install from PyPI:

.. code-block:: bash

   pip install astro-brutus

Development Install
-------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/joshspeagle/brutus.git
   cd brutus
   pip install -e ".[dev]"

Windows Users - Important Note
-------------------------------

⚠️ **Windows Compatibility**: Due to the ``healpy`` dependency (required for dust mapping), brutus does not work reliably on native Windows. **Windows users should install and run brutus in WSL (Windows Subsystem for Linux)**.

Alternative Windows installation options:

- **WSL (Recommended)**: Install Ubuntu or another Linux distribution via WSL and use the standard installation
- **Conda**: Try ``conda install -c conda-forge astro-brutus`` which may have pre-compiled Windows wheels
- **Docker**: Use a Linux-based Docker container

Conda Installation
------------------

If you use conda, you may be able to install from conda-forge (availability varies):

.. code-block:: bash

   conda install -c conda-forge astro-brutus

.. note::
   If the conda-forge package is unavailable, use ``pip install astro-brutus`` instead.

Dependencies
------------

Core dependencies that will be automatically installed:

- ``numpy`` (≥1.19) - Numerical computing
- ``scipy`` (≥1.6) - Scientific computing
- ``matplotlib`` (≥3.3) - Plotting
- ``h5py`` (≥3.0) - HDF5 file support
- ``healpy`` (≥1.14) - HEALPix utilities for incorporating dust maps
- ``numba`` (≥0.53) - Just-in-time compilation for performance
- ``pooch`` (≥1.4) - Data downloading and management
- ``tqdm`` (≥4.50) - Progress bars and live tracking

Data Files
----------

.. warning::

   **You must download data files before using brutus.** The package will not work without them.

After installing brutus, download the required model data:

.. list-table:: Data Download Checklist
   :header-rows: 1
   :widths: 15 15 50 20

   * - Data
     - Size
     - Purpose
     - Required?
   * - ``fetch_grids()``
     - 1-5 GB
     - Pre-computed stellar model grids for fast fitting
     - **Yes**
   * - ``fetch_isos()``
     - ~100 MB
     - Isochrone tables for stellar evolution
     - **Yes**
   * - ``fetch_dustmaps()``
     - ~1 GB
     - 3D dust maps (Bayestar) for extinction priors
     - Optional

**Minimum setup** (required for all fitting):

.. code-block:: python

   from brutus import fetch_grids, fetch_isos
   fetch_grids()   # Required - stellar model grids
   fetch_isos()    # Required - isochrone data

**Full setup** (adds 3D dust priors):

.. code-block:: python

   from brutus import fetch_grids, fetch_isos, fetch_dustmaps
   fetch_grids()
   fetch_isos()
   fetch_dustmaps()  # Optional - needed for dust map priors

Files are cached in your user data directory (``~/.brutus/`` or platform equivalent) and only downloaded once. Total disk space needed: **2-7 GB** depending on options.

Testing the Installation
-------------------------

To verify your installation works correctly:

.. code-block:: python

   import brutus
   print(f"brutus version: {brutus.__version__}")

   # Test core functionality
   from brutus import Isochrone
   print("Installation successful!")
