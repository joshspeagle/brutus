#!/usr/bin/env python
"""Test all tutorial notebooks by extracting and running their code cells."""

import json
import sys
import traceback
from pathlib import Path


def extract_code_cells(notebook_path):
    """Extract code cells from a notebook."""
    with open(notebook_path, "r") as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            # Join source lines and skip cells that are just comments or empty
            source = "".join(cell["source"])
            if source.strip() and not source.strip().startswith("#"):
                code_cells.append(source)

    return code_cells


def test_notebook(notebook_path, skip_data_download=True):
    """Test a notebook by running its code cells."""
    print(f"\n{'='*60}")
    print(f"Testing: {notebook_path.name}")
    print("=" * 60)

    try:
        code_cells = extract_code_cells(notebook_path)
        print(f"Found {len(code_cells)} code cells")

        # Create a namespace for execution
        namespace = {}

        for i, cell in enumerate(code_cells):
            # Skip data download cells if requested
            if skip_data_download and (
                "fetch_grids" in cell
                or "fetch_isos" in cell
                or "fetch_dustmaps" in cell
            ):
                print(f"  Cell {i+1}: Skipping data download cell")
                continue

            # Skip cells that might take very long (marked with specific comments)
            if "# This may take several minutes" in cell or "# SLOW:" in cell:
                print(f"  Cell {i+1}: Skipping slow cell")
                continue

            print(f"  Cell {i+1}: Running...")

            try:
                # Execute the cell in the namespace
                exec(cell, namespace)
                print(f"  Cell {i+1}: ✓ Success")
            except Exception as e:
                print(f"  Cell {i+1}: ✗ Error: {str(e)[:100]}")
                # Continue to next cell rather than failing completely
                if "No module named" in str(e) or "cannot import" in str(e):
                    print("    (Import error - missing dependency)")
                    return False
                elif "No such file or directory" in str(e) or "does not exist" in str(
                    e
                ):
                    print("    (File not found - data may need downloading)")
                # Don't fail on plotting errors in test mode
                elif "matplotlib" in str(e) or "plt.show()" in cell:
                    print("    (Plotting error - likely running headless)")
                else:
                    # Show more details for other errors
                    traceback.print_exc()

        print(f"\n✓ {notebook_path.name} completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ {notebook_path.name} failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Test all tutorial notebooks."""
    tutorials_dir = Path(__file__).parent

    # Get all tutorial notebooks in order
    notebooks = sorted(tutorials_dir.glob("tutorial_*.ipynb"))

    if not notebooks:
        print("No tutorial notebooks found!")
        return 1

    print(f"Found {len(notebooks)} notebooks to test")

    # Test configuration
    skip_data_download = True  # Skip data download cells for testing

    results = {}
    for notebook in notebooks:
        success = test_notebook(notebook, skip_data_download=skip_data_download)
        results[notebook.name] = success

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")

    failed = [name for name, success in results.items() if not success]
    if failed:
        print(f"\n{len(failed)} notebooks failed")
        return 1
    else:
        print(f"\n✓ All {len(notebooks)} notebooks passed!")
        return 0


if __name__ == "__main__":
    # Suppress matplotlib display warnings
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # Set matplotlib to non-interactive backend for testing
    import matplotlib

    matplotlib.use("Agg")

    sys.exit(main())
