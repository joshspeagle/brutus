Installation
============

Requirements
------------

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL (see Windows note below)

Quick Install
-------------

Install the latest stable release from `PyPI <https://pypi.org/project/astro-brutus/>`_:

.. code-block:: bash

   pip install astro-brutus

Development Install
-------------------

Install the development version (v\ |version|) directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/joshspeagle/brutus.git

Or clone for local development:

.. code-block:: bash

   git clone https://github.com/joshspeagle/brutus.git
   cd brutus
   pip install -e ".[dev]"

Windows Users
-------------

.. warning::

   Due to the ``healpy`` dependency (required for dust mapping), brutus does not
   work on native Windows. **Windows users must use WSL (Windows Subsystem for Linux)**.

To install on Windows:

1. Install `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ with Ubuntu or another Linux distribution
2. Open a WSL terminal and follow the standard installation instructions above

Alternatively, use a Linux-based Docker container.

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
   :widths: 20 14 46 20

   * - Function
     - Size
     - Purpose
     - Required?
   * - ``fetch_grids()``
     - ~750 MB
     - Pre-computed stellar model grids for fast fitting
     - **Yes**
   * - ``fetch_isos()``
     - ~250 MB
     - Isochrone tables for stellar populations
     - **Yes**
   * - ``fetch_offsets()``
     - <1 KB
     - Photometric calibration offsets for improved accuracy
     - Recommended
   * - ``fetch_dustmaps()``
     - ~2 GB
     - 3D dust maps (Bayestar) for extinction priors
     - Optional

**Recommended setup**:

.. code-block:: python

   from brutus.data import fetch_grids, fetch_isos, fetch_offsets

   # Download stellar model grid (~750 MB)
   grid_path = fetch_grids(grid='mist_v9')

   # Download isochrone tables (~250 MB)
   iso_path = fetch_isos(iso='MIST_1.2_vvcrit0.0')

   # Download photometric offsets (recommended for fitting)
   offset_path = fetch_offsets(grid='mist_v9')

**With 3D dust maps** (for extinction priors):

.. code-block:: python

   from brutus.data import fetch_dustmaps

   # Download Bayestar dust maps (~2 GB)
   dustmap_path = fetch_dustmaps(dustmap='bayestar19')

Photometric Offsets
^^^^^^^^^^^^^^^^^^^

The photometric offsets correct for systematic differences between observed and
model photometry. These are derived from well-characterized calibration stars
and significantly improve fitting accuracy. While not strictly required, using
offsets is strongly recommended for reliable results.

Files are cached in your user data directory and only downloaded once.
Total disk space needed: **~1 GB** minimum, **~3 GB** with dust maps.

Testing the Installation
------------------------

To verify your installation works correctly:

.. code-block:: python

   import brutus
   print(f"brutus version: {brutus.__version__}")

   # Test core imports
   from brutus.core import Isochrone, StarGrid
   from brutus.analysis import BruteForce
   print("Installation successful!")
