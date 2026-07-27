# Sphinx configuration for the DMK documentation.

import os

project = "DMK"
copyright = "2025, The Simons Foundation, Inc."
author = "Leslie Greengard, Shidong Jiang, Robert Blackwell"

# Displayed version: hardcoded default, overridden by the git tag on Read the Docs
# tag builds
version = release = "1.0"
if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag":
    version = release = os.environ.get("READTHEDOCS_VERSION", version).lstrip("v")

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
