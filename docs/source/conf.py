"""Sphinx configuration for the brutus documentation."""

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../src"))


# -- Copy tutorial notebooks into docs at build time -------------------------


def _copy_tutorials():
    """Copy tutorial notebooks from tutorials/ into docs/source/tutorials/."""
    source_dir = Path(__file__).resolve().parent.parent.parent / "tutorials"
    dest_dir = Path(__file__).resolve().parent / "tutorials"
    dest_dir.mkdir(exist_ok=True)

    for nb in sorted(source_dir.glob("tutorial_*.ipynb")):
        dest = dest_dir / nb.name
        shutil.copy2(nb, dest)


_copy_tutorials()

# -- Project information -----------------------------------------------------

project = "brutus"
copyright = "2025, Joshua S. Speagle"
author = "Joshua S. Speagle"

# Auto-detect version from package metadata or pyproject.toml
try:
    from importlib.metadata import version as get_version

    release = get_version("astro-brutus")
except Exception:
    # Fallback: parse pyproject.toml if package not installed
    import tomllib

    with open("../../pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    release = pyproject["project"]["version"]

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_nb",
    "sphinxcontrib.bibtex",
]

# myst-nb configuration
nb_execution_mode = "off"  # Use pre-rendered outputs, don't execute during build
suppress_warnings = ["mystnb.unknown_mime_type"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "brutus"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["brutus.css"]
html_last_updated_fmt = "%b %d, %Y"

# The palette comes from the logo: gold, sand, amber, copper and deep red.
html_theme_options = {
    "light_logo": "brutus-logo-light.svg",
    "dark_logo": "brutus-logo-dark.svg",
    "sidebar_hide_name": True,
    # No view or edit buttons: the tutorial pages are notebooks copied in at
    # build time, so their source links would point to missing files.
    "top_of_page_buttons": [],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/joshspeagle/brutus",
            "html": '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>',
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/astro-brutus/",
            "html": "PyPI",
            "class": "",
        },
    ],
    "light_css_variables": {
        "color-brand-primary": "#7C0008",
        "color-brand-content": "#A3401A",
        "color-brand-visited": "#8A3A1E",
        "color-background-primary": "#FFFDF9",
        "color-background-secondary": "#FBF4EA",
        "color-background-hover": "#F5E6D3",
        "color-highlighted-background": "#FBE7C4",
        "color-card-border": "#ECD9C2",
        "color-card-background": "#FFFDF9",
        "sd-color-primary": "#7C0008",
        "sd-color-primary-highlight": "#A3401A",
        "sd-color-primary-text": "#FFFFFF",
        "brutus-gold": "#F6C35F",
        "brutus-amber": "#D58A4C",
        "brutus-copper": "#B5562B",
        "brutus-red": "#7C0008",
        "brutus-dot": "rgba(124, 0, 8, 0.16)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FCD07C",
        "color-brand-content": "#ECB068",
        "color-brand-visited": "#D89054",
        "color-background-primary": "#15110F",
        "color-background-secondary": "#1D1714",
        "color-background-hover": "#2A201B",
        "color-highlighted-background": "#4A2E1A",
        "color-card-border": "#3A2B23",
        "color-card-background": "#15110F",
        "sd-color-primary": "#FCD07C",
        "sd-color-primary-highlight": "#ECB068",
        "sd-color-primary-text": "#15110F",
        "brutus-gold": "#FCD07C",
        "brutus-amber": "#D89054",
        "brutus-copper": "#E08A55",
        "brutus-red": "#D9544D",
        "brutus-dot": "rgba(252, 208, 124, 0.13)",
    },
}

# -- Extension configuration -------------------------------------------------

# autodoc options
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False

# autosummary settings
autosummary_generate = True

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest/", None),
}

# sphinx-copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
