# Sphinx configuration for the DMK documentation.

project = "DMK"
copyright = "2025, The Simons Foundation, Inc."
author = "Leslie Greengard, Shidong Jiang, Robert Blackwell"

# Hardcoded for now; CMake has no project(dmk VERSION ...) yet (see RELEASE_1.0.md).
version = "1.0"
release = "1.0"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "breathe",
]

master_doc = "index"
source_suffix = {".rst": "restructuredtext"}

# Prevent RST literal blocks from being mis-highlighted as Python.
highlight_language = "none"

html_theme = "sphinx_rtd_theme"

# Breathe pulls the Doxygen-generated XML (see docs/Doxyfile) into Sphinx.
breathe_projects = {"dmk": "doxygen/xml"}
breathe_default_project = "dmk"
