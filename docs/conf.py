# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "nyx"
copyright = "2026, Gerrit Roellinghoff"
author = "Gerrit Roellinghoff"

# Version handling
try:
    from nyx import __version__

    release = __version__
except ImportError:
    release = "dev"

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_automodapi.automodapi",
    "sphinx_design",
    "myst_nb",
]

# -- Options for myst-nb -----------------------------------------------------

# Don't execute notebooks during build (they should be pre-executed)
nb_execution_mode = "off"

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

# myst-parser configuration
myst_enable_extensions = [
    "dollarmath",  # Enable $ and $$ for math
    "colon_fence",  # Enable ::: fences
]

nb_render_markdown_format = "myst"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# -- Options for autodoc -----------------------------------------------------

autodoc_typehints = "none"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for automodapi --------------------------------------------------

numpydoc_show_class_members = False

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/GerritRo/nyx",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Options for Napoleon extension ------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True
