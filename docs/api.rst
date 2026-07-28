C API reference
===============

The public interface is the C header ``include/dmk.h``. This page is generated from that
header via Doxygen and Breathe, so it stays in sync with the source.

Every double-precision entry point has a single-precision counterpart suffixed with ``f``
(e.g. ``pdmk_tree_create`` / ``pdmk_tree_createf``). Only the double-precision forms are
listed below; the ``f`` forms take the corresponding ``float`` arrays.

Parameters
----------

.. doxygenstruct:: pdmk_params
   :members:
   :undoc-members:

Enumerations
------------

.. doxygenenum:: dmk_ikernel

.. doxygenenum:: dmk_eval_type

.. doxygenenum:: dmk_error

.. doxygenenum:: dmk_log_level

Setup and error handling
------------------------

.. doxygenfunction:: pdmk_init_default_params

.. doxygenfunction:: pdmk_last_error_message

Tree lifecycle
--------------

.. doxygenfunction:: pdmk_tree_create

.. doxygenfunction:: pdmk_tree_eval

.. doxygenfunction:: pdmk_tree_update_charges

.. doxygenfunction:: pdmk_tree_destroy

.. doxygenfunction:: pdmk

Profiling
---------

.. doxygenfunction:: pdmk_print_profile_data

ESP: Ewald summation with prolates
----------------------------------

DMK includes an experimental periodic (and free-space) electrostatics solver based on
prolate spheroidal wave functions. Particles lie in the cubic box
:math:`[-L/2,\, L/2)^{n}`.

.. doxygenstruct:: pdmk_esp_params
   :members:
   :undoc-members:

.. doxygenfunction:: pdmk_esp_plan_create

.. doxygenfunction:: pdmk_esp_eval

.. doxygenfunction:: pdmk_esp_plan_destroy

.. doxygenfunction:: pdmk_esp
