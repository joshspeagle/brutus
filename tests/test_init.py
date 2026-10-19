"""Tests of the package-level ``brutus`` namespace."""

import subprocess
import sys


def test_import_brutus_raises_when_a_dependency_is_missing():
    """A missing dependency raises an error that names it, not a hidden warning."""
    code = "import sys; sys.modules['healpy'] = None; import brutus"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "healpy" in result.stderr
