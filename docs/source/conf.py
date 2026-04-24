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
    "nbsphinx",
]

extensions += ["sphinx_design"]
extensions += ["sphinx_gallery.gen_gallery"]

nbsphinx_execute = "never"

examples_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../example/documentation_examples"))

sphinx_gallery_conf = {
    "examples_dirs": examples_path,      # where your scripts live
    "gallery_dirs": "auto_examples",     # generated site
    "filename_pattern": r"\.py",
    "run_stale_examples": True
}

templates_path = ['_templates']
exclude_patterns = [
    "auto_examples/**/*.ipynb",
    "auto_examples/**/*.py",
    "auto_examples/**/*.py.md5",
    "auto_examples/**/*.codeobj.json",
    "auto_examples/**/*.zip",
    "api/_api_stubs.rst",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "imported-members": False,
    "show-inheritance": True,
    "exclude-members": "__dict__,__weakref__,__module__,__pydantic_core_schema__,__pydantic_validator__,__pydantic_serializer__",
}

autosummary_generate = True
autosummary_imported_members = False
add_module_names = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "tracts",
    },
    "github_url": "https://github.com/gravellab/tracts",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_depth": 2,
    "collapse_navigation": False,
}

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

def skip_abstract_methods(app, what, name, obj, skip, options):
    import inspect
    if inspect.isfunction(obj) and getattr(obj, "__isabstractmethod__", False):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_abstract_methods)


