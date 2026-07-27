# Makefile for the DMK Sphinx documentation.
# `make html` runs Doxygen first so Breathe has fresh XML to consume.

PYTHON      = python
SPHINXBUILD = $(PYTHON) -m sphinx
SPHINXOPTS  =
BUILDDIR    = _build

.PHONY: help doxygen html clean

help:
	@echo "Targets: doxygen, html, clean"

doxygen:
	doxygen Doxyfile

html: doxygen
	$(SPHINXBUILD) -b html $(SPHINXOPTS) . $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	rm -rf $(BUILDDIR) doxygen
