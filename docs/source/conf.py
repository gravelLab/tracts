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
release = '2.0.0'

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
    "run_stale_examples": False,
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

# Without this, autodoc/sphinx_autodoc_typehints resolve "npt.ArrayLike" to its full underlying
# Union definition (a wall of Buffer/_SupportsArray/_NestedSequence/... types) instead of showing
# it as "ArrayLike". Requires the annotated modules to use postponed evaluation of annotations
# (`from __future__ import annotations`) so the alias name survives as text for this to match.
autodoc_type_aliases = {
    "npt.ArrayLike": "numpy.typing.ArrayLike",
    "ArrayLike": "numpy.typing.ArrayLike",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# Renders docstring-declared types (e.g. a bare "Returns"/"Parameters" type line) the same way as
# real type annotations (monospaced, cross-referenced), instead of as plain text. Note: Napoleon's
# type tokenizer only avoids splitting on a comma when it is *not* followed by a space, so compound
# return types with multiple elements (tuples, dicts, etc.) must be written without a space after
# each internal comma, e.g. "tuple[int,float]" not "tuple[int, float]", or they get garbled into
# multiple broken fragments.
napoleon_preprocess_types = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo/tracts-logo-light.svg",
        "image_dark": "_static/logo/tracts-logo-dark.svg",
        "alt_text": "tracts",
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

html_favicon = "_static/logo/tracts-favicon.ico"

html_css_files = [
    "custom.css",
]

def skip_private_and_abstract_members(app, what, name, obj, skip, options):
    import inspect
    # Single-underscore-prefixed names (e.g. "_bin_centers") are internal helpers and are never
    # documented online; dunder names (e.g. "__init__") are unaffected and still handled by the
    # usual autodoc_default_options/exclude-members logic below.
    if name.startswith("_") and not name.startswith("__"):
        return True
    if inspect.isfunction(obj) and getattr(obj, "__isabstractmethod__", False):
        return True
    return None

def skip_duplicate_pydantic_and_enum_members(app, what, name, obj, skip, options):
    # Pydantic model fields and Enum members are real class attributes, so autodoc documents them
    # a second time as bare "name: type" entries, on top of the class's own numpydoc "Attributes"
    # section (rendered by napoleon). Every pydantic/Enum class in this codebase already documents
    # its fields/members there, so the auto-generated duplicates are skipped here instead of being
    # hand-listed (and kept up to date) per class in the autosummary class template.
    if what != "class":
        return skip
    import dataclasses
    import importlib
    from enum import Enum
    from pydantic import BaseModel
    modname = app.env.temp_data.get("autodoc:module")
    clsname = app.env.temp_data.get("autodoc:class")
    if not modname or not clsname:
        return skip
    try:
        cls = getattr(importlib.import_module(modname), clsname)
    except Exception:
        return skip
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        if name == "model_config" or name in cls.model_fields:
            return True
    elif isinstance(cls, type) and issubclass(cls, Enum):
        if name in cls.__members__:
            return True
    elif dataclasses.is_dataclass(cls):
        if name in {field.name for field in dataclasses.fields(cls)}:
            return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_private_and_abstract_members)
    app.connect("autodoc-skip-member", skip_duplicate_pydantic_and_enum_members)


