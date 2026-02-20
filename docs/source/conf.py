# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

project = 'tracts'
copyright = '2025, Javier González-Delgado, Andrii Serdiuk, Victor Krim-Yee and Simon Gravel'
author = 'Javier González-Delgado, Andrii Serdiuk, Victor Krim-Yee and Simon Gravel'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "members": True,            # include all functions and classes
    "undoc-members": True,      # include members without docstrings
    "inherited-members": True,  # include inherited methods
    "show-inheritance": True,   # show class inheritance
    "special-members": "__init__",  # include constructor
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
