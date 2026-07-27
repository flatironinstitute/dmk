Quickstart
==========

DMK is used through the C API declared in ``include/dmk.h``. The typical workflow is:

#. fill a :cpp:struct:`pdmk_params` with the library defaults and override what you need;
#. build a tree from the sources (and, optionally, separate targets);
#. evaluate the potential;
#. destroy the tree.

Every entry point has a double-precision form and a single-precision form suffixed with
``f`` (e.g. :cpp:func:`pdmk_tree_create` / ``pdmk_tree_createf``).

A minimal example
-----------------

The following builds a tree over ``n_src`` sources in 3D and evaluates the Laplace potential
at the sources. Coordinates are interleaved (``x0, y0, z0, x1, y1, z1, ...``).

.. code-block:: cpp

   #include <dmk.h>
   #include <mpi.h>
   #include <stdexcept>
   #include <vector>

   int main(int argc, char *argv[]) {
       int provided;
       MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

       const int n_dim = 3;
       const int n_src = 1000000;
       const int n_trg = 0; // evaluate at the sources themselves

       // ... fill r_src (n_dim * n_src) and charges (n_src) with your data ...
       std::vector<double> r_src, charges, normal, r_trg;
       std::vector<double> pot_src(n_src), pot_trg(n_trg);

       pdmk_params params;
       pdmk_init_default_params(&params);
       params.n_dim = n_dim;
       params.eps = 1e-3;            // target accuracy (three digits)
       params.kernel = DMK_LAPLACE;
       params.eval_src = DMK_POTENTIAL;
       params.eval_trg = DMK_POTENTIAL;

       pdmk_tree tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_src,
                                         r_src.data(), charges.data(), normal.data(),
                                         n_trg, r_trg.data());
       if (!tree)
           throw std::runtime_error(pdmk_last_error_message());

       pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());

       pdmk_tree_destroy(tree);
       MPI_Finalize();
       return 0;
   }

The ``normal`` array is only read by the Stresslet kernel (one orientation vector per
source); for other kernels it may be empty or ``NULL``.

Reusing a tree
--------------

The tree geometry depends only on the point locations, so if only the charges change you can
re-evaluate without rebuilding:

.. code-block:: cpp

   pdmk_tree_update_charges(tree, new_charges.data(), normal.data());
   pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());

One-shot evaluation
-------------------

For a single evaluation, :cpp:func:`pdmk` combines create, eval, and destroy into one call:

.. code-block:: cpp

   pdmk(MPI_COMM_WORLD, params, n_src, r_src.data(), charges.data(), normal.data(),
        n_trg, r_trg.data(), pot_src.data(), pot_trg.data());

A complete, runnable version of this example (with random test data and timing) is in
``examples/main.cpp``.
