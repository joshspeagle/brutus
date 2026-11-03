# Dust Module (`brutus.dust`)

`brutus.dust` queries three-dimensional interstellar dust maps on multi-resolution HEALPix grids and maps sky coordinates to HEALPix pixels.

**Key Components:**

- **Dust Map Interface (`DustMap`):** Abstract base class supporting queries by `astropy.coordinates.SkyCoord` (`query`), Galactic coordinates (`query_gal`), or equatorial coordinates (`query_equ`).
- **Bayestar 3D Dust Map (`Bayestar`):** Queries the {cite:t}`green2019` Bayestar19 dust map (`bayestar2019_v1.h5`) across the Pan-STARRS 1 footprint ($\delta > -30^\circ$), returning the distance grid (in kpc) and the mean and standard deviation of $A_V$ (in mag) along each sightline while applying per-pixel convergence and distance-reliability masks by default.
- **Coordinate Lookup (`lb2pix`):** Converts Galactic $(\ell, b)$ coordinates in degrees to nested or ring HEALPix pixel indices.

```{eval-rst}
.. currentmodule:: brutus.dust
```

## Dust Maps

```{eval-rst}
.. autoclass:: DustMap
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: Bayestar
   :members:
   :undoc-members:
   :show-inheritance:
```

## Coordinates

```{eval-rst}
.. autofunction:: lb2pix
```

## Submodules

```{eval-rst}
.. automodule:: brutus.dust.maps
   :no-members:
```

```{eval-rst}
.. automodule:: brutus.dust.extinction
   :no-members:
```
