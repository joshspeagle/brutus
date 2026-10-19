#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Additional tests to improve galactic.py coverage.

These tests specifically target the uncovered lines in galactic.py
to achieve better test coverage.
"""

import numpy as np

# pytest used by test framework
from astropy.coordinates import SkyCoord

from brutus.priors.galactic import logp_age_from_feh, logp_galactic_structure


class TestGalacticCoverage:
    """Tests to improve galactic.py code coverage."""

    def test_galactic_structure_tuple_coordinates(self):
        """Test galactic structure with tuple coordinates (not SkyCoord)."""
        distances = np.array([1.0, 2.0])
        coord_tuple = (180.0, 0.0)  # (l, b) in degrees

        logp = logp_galactic_structure(distances, coord_tuple)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_labels_feh(self):
        """Test galactic structure with metallicity labels."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with metallicity data
        labels = np.array([(0.0,), (-0.5,)], dtype=[("feh", "f4")])

        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_labels_age(self):
        """Test galactic structure with age labels."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with log(age) data
        labels = np.array(
            [(9.0,), (9.5,)], dtype=[("loga", "f4")]
        )  # log10(age in years)

        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_labels_both(self):
        """Test galactic structure with both metallicity and age labels."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with both feh and loga
        labels = np.array(
            [(0.0, 9.0), (-0.5, 9.5)], dtype=[("feh", "f4"), ("loga", "f4")]
        )

        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_with_invalid_labels(self):
        """Test galactic structure with invalid/missing label fields."""
        distances = np.array([1.0, 2.0])
        coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")

        # Create structured array with different field name
        labels = np.array([(0.0,), (-0.5,)], dtype=[("metallicity", "f4")])

        # Should handle missing fields gracefully
        logp = logp_galactic_structure(distances, coord, labels=labels)

        assert np.all(np.isfinite(logp))
        assert len(logp) == len(distances)

    def test_galactic_structure_components_with_labels(self):
        """Test galactic structure returning components with labels."""
        distances = np.array([1.0])
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")

        # Create labels with both metallicity and age
        labels = np.array([(0.0, 9.0)], dtype=[("feh", "f4"), ("loga", "f4")])

        logp, components = logp_galactic_structure(
            distances, coord, labels=labels, return_components=True
        )

        # Should have all component types
        assert "number_density" in components
        assert "feh" in components
        assert "age" in components

        # Each component should have three subcomponents (thin, thick, halo)
        assert len(components["number_density"]) == 3
        assert len(components["feh"]) == 3
        assert len(components["age"]) == 3

    def test_logp_age_from_feh_parameter_matching(self):
        """Test age-metallicity function with correct parameter names."""
        ages = np.array([5.0, 8.0])

        # Check actual function signature parameters
        logp_default = logp_age_from_feh(ages)
        logp_custom = logp_age_from_feh(ages, feh_mean=-0.1, max_age=12.0, min_age=0.5)

        # Should produce different results
        assert not np.allclose(logp_default, logp_custom)
        assert np.all(np.isfinite(logp_default))
        assert np.all(np.isfinite(logp_custom))


class TestLegacyImportFallback:
    """Test legacy import fallback (difficult to test without modifying imports)."""

    def test_scipy_import_works(self):
        """Test that scipy.special.logsumexp is available."""
        # This implicitly tests that the import succeeded
        from brutus.priors.galactic import logp_galactic_structure

        distances = np.array([1.0, 2.0, 3.0])
        coord = SkyCoord(ra=0.0, dec=0.0, unit="deg")

        logp = logp_galactic_structure(distances, coord)

        # The fact that this works means logsumexp was imported correctly
        assert np.all(np.isfinite(logp))
        assert len(logp) == 3
