# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

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

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# extensions.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "numpydoc",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

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
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# numpydoc settings
numpydoc_show_class_members = False

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

# -- HTML theme options ------------------------------------------------------

html_theme_options = {
    "repository_url": "https://github.com/joshspeagle/brutus",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "repository_branch": "master",
    "path_to_docs": "docs/source",
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/brutus_logo.png",
        "image_dark": "_static/brutus_logo.png",
        "text": "brutus",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/astro-brutus/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
}

# HTML context
html_context = {
    "github_user": "joshspeagle",
    "github_repo": "brutus",
    "github_version": "master",
    "doc_path": "docs/source",
}

html_title = "brutus Documentation"
html_short_title = "brutus"
html_favicon = None

# Additional HTML options
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
# sphinx_book_theme handles sidebars automatically
html_additional_pages = {}
html_domain_indices = True
html_use_index = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# LaTeX output
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
