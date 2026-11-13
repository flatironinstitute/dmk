Quickstart
==========

DMK is used through the C API declared in ``dmk.h``. The typical workflow is:

#. fill a :cpp:struct:`pdmk_params` with the library defaults and override what you need;
#. build a tree from the sources (and, optionally, separate targets);
#. evaluate the potential;
#. destroy the tree.

Every entry point has a double-precision form and a single-precision form suffixed with
``f`` (e.g. :cpp:func:`pdmk_tree_create` / ``pdmk_tree_createf``). ``pdmk_tree_create``
returns ``NULL`` on failure; the other calls return a :cpp:enum:`dmk_error`, which is
``DMK_SUCCESS`` (zero) on success. Either way, ``pdmk_last_error_message()`` returns the
detail for the last failing call on the calling thread.

A minimal example
-----------------

The following builds a tree over ``n_src`` sources in 3D and evaluates the Laplace potential
at the sources. Coordinates are interleaved (``x0, y0, z0, x1, y1, z1, ...``), as is the
output: one value per point for ``DMK_POTENTIAL``, ``1 + n_dim`` for ``DMK_POTENTIAL_GRAD``,
and ``n_dim`` for ``DMK_VELOCITY``.

.. code-block:: c

   #include <dmk.h>

   #include <mpi.h>
   #include <stdio.h>
   #include <stdlib.h>

   int main(int argc, char **argv) {
       MPI_Init(&argc, &argv);

       const int n_dim = 3;
       const int n_src = 1000000;

       double *r_src = malloc(sizeof(double) * n_dim * n_src);
       double *charge = malloc(sizeof(double) * n_src);
       double *pot_src = malloc(sizeof(double) * n_src);

       /* ... fill r_src (n_dim * n_src, interleaved) and charge (n_src) with your data ... */

       pdmk_params params;
       pdmk_init_default_params(&params);
       params.n_dim = n_dim;
       params.eps = 1e-3; /* target accuracy: three digits */
       params.kernel = DMK_LAPLACE;
       params.eval_src = DMK_POTENTIAL;
       params.eval_trg = DMK_POTENTIAL;

       /* No separate targets, so n_trg is 0 and r_trg is NULL. */
       pdmk_tree tree =
           pdmk_tree_create(MPI_COMM_WORLD, params, n_src, r_src, charge, NULL, 0, NULL);
       if (!tree) {
           fprintf(stderr, "pdmk_tree_create: %s\n", pdmk_last_error_message());
           return 1;
       }

       dmk_error err = pdmk_tree_eval(tree, pot_src, NULL);
       if (err != DMK_SUCCESS) {
           fprintf(stderr, "pdmk_tree_eval: %s\n", pdmk_last_error_message());
           pdmk_tree_destroy(tree);
           return 1;
       }

       pdmk_tree_destroy(tree);
       free(r_src);
       free(charge);
       free(pot_src);
       MPI_Finalize();
       return 0;
   }

The ``normal`` array is only read by the Stresslet kernel (one orientation vector per
source); for every other kernel it may be ``NULL``.

``pot_src`` and ``pot_trg`` must point at buffers large enough for their own point set.
Unlike :cpp:func:`pdmk_direct`, the tree path does not read a null output pointer as "skip
this point set"; ``NULL`` is only safe for a point set that is empty, as ``pot_trg`` is above
with ``n_trg = 0``.

Reusing a tree
--------------

The tree geometry depends only on the point locations, so if only the charges change you can
re-evaluate without rebuilding:

.. code-block:: c

   dmk_error err = pdmk_tree_update_charges(tree, new_charge, NULL);
   if (err == DMK_SUCCESS)
       err = pdmk_tree_eval(tree, pot_src, NULL);

One-shot evaluation
-------------------

For a single evaluation, :cpp:func:`pdmk` combines create, eval, and destroy into one call:

.. code-block:: c

   dmk_error err = pdmk(MPI_COMM_WORLD, params, n_src, r_src, charge, NULL,
                        0, NULL, pot_src, NULL);

Complete examples
-----------------

``examples/example_c.c`` is a compilable C program covering the same flow -- create, eval,
``pdmk_tree_update_charges``, eval again, destroy -- for the Stokeslet. It is built as the
``example_c`` target when examples are enabled. ``examples/main.cpp`` is the C++ counterpart,
with random test data and timing.
