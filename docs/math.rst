Mathematical background
=======================

DMK evaluates convolutions of a radially symmetric, non-oscillatory kernel :math:`K` against
a set of sources. In the discrete case,

.. math::

   u(x_i) = \sum_{j=1}^{N_S} K(x_i, y_j)\, \rho_j, \quad i = 1, \ldots, N_T,

where :math:`y_j` are the source locations, :math:`\rho_j` the source strengths (charges),
and :math:`x_i` the target locations. Sources and targets may coincide.

The algorithm smooths the kernel at the coarsest grid level, applies a sequence of
corrections at successively finer scales, and finishes with direct summation once the
residual interaction is local. Each scale is diagonalized by a short Fourier transform,
which unifies the fast multipole method, Ewald summation, and multilevel summation. For
continuous source distributions, the finest-level interaction is further accelerated by
approximating the kernel as a sum of Gaussians with a localized remainder. See the
:doc:`references <refs>` for the full derivation.

Supported kernels
-----------------

The kernel is selected with the ``dmk_ikernel`` enum (see ``include/dmk.h``). Precise
definitions, including dimension-dependent normalizations, are given in the DMK paper.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constant
     - Kernel
   * - ``DMK_YUKAWA``
     - Screened Coulomb (Yukawa), :math:`e^{-\lambda r}/r`; the screening parameter
       :math:`\lambda` is passed as ``fparam``.
   * - ``DMK_LAPLACE``
     - Laplace / Coulomb: :math:`1/r` in 3D, :math:`\tfrac{1}{2}\log r^2` in 2D.
   * - ``DMK_SQRT_LAPLACE``
     - Power-law kernel (dimension-dependent normalization; see the paper).
   * - ``DMK_STOKESLET``
     - Stokeslet, the Green's function of incompressible Stokes flow.
   * - ``DMK_STRESSLET``
     - Stresslet, the Stokes stress tensor; requires a per-source orientation (``normal``).
   * - ``DMK_LAPLACE_DIPOLE``
     - Laplace dipole kernel.

Evaluation types
----------------

The quantities computed at sources and targets are selected independently with the
``dmk_eval_type`` enum, via the ``eval_src`` and ``eval_trg`` parameters.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Constant
     - Quantity
   * - ``DMK_POTENTIAL``
     - Potential :math:`\phi` only.
   * - ``DMK_POTENTIAL_GRAD``
     - Potential and gradient :math:`\nabla\phi`.
   * - ``DMK_POTENTIAL_GRAD_HESSIAN``
     - Potential, gradient, and Hessian.
   * - ``DMK_VELOCITY``
     - Velocity field (Stokes kernels).
   * - ``DMK_VELOCITY_PRESSURE``
     - Velocity and pressure (Stokes kernels).
