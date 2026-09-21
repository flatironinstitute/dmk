Mathematical background
=======================

DMK evaluates convolutions of a radially symmetric, non-oscillatory kernel :math:`K` against
a set of sources. In the discrete case,

.. math::

   u(x_i) = \sum_{j=1}^{N_S} K(x_i, y_j)\, \rho_j, \quad i = 1, \ldots, N_T,

where :math:`y_j` are the source locations, :math:`\rho_j` the source strengths (charges),
and :math:`x_i` the target locations. Sources and targets may coincide.

The key idea of the telescoping DMK algorithm is that the above sum can be replaced by a
multi-level Ewald-like decomposition with an additional intermediate term.  For example, the 3D
Laplace kernel

.. math::

   u(r) \approx W_0(r) + \sum_{l=0}^{L-1} D_l(r) + R_L(r),
   \qquad L = 0,...,L_\max

A *very* rough sketch of the current point DMK algorithm from the equation above:

1. Build a quad/octree out of input point data
2. Approximate the field of the leaf boxes by moving them to a chebyshev grid of proxy charges
3. Translate those fields up the tree via tensor-product transforms
4. Use those fields to evaluate the :math:`W_(r)` potential at the root on the proxy charges
5. Translate those fields down usign :math:`D_l(r)`, translating the fields of neighbor boxes
   into each boxes via planewaves
6. Apply the proxy field to the points at the lowest level
7. Evaluate residual kernels :math:`R_L(r)` between boxes and neighboring boxes at the lowest level.

See the :doc:`references <refs>` for the full derivation.

Notation
--------

Throughout, :math:`y_j` is a source location, :math:`x_i` a target location, and

.. math::

   R = x_i - y_j, \qquad
   r = |R| = \Big(\textstyle\sum_{k=1}^{d} (x_{i,k} - y_{j,k})^2\Big)^{1/2}.

Pairs with :math:`r = 0` are skipped, so coincident source/target points contribute nothing. No
:math:`4\pi` or viscosity normalization is folded into any kernel: the sums below are exactly
what DMK returns, and the caller is responsible for any physical prefactor.

Kernels
-------

The kernel is selected with the ``dmk_ikernel`` enum (see ``include/dmk.h``). The dimensions
listed here are those for which the kernel is defined; see :doc:`features` for which of them
each compute path (tree, direct, GPU) actually supports.

Laplace (``DMK_LAPLACE``)
~~~~~~~~~~~~~~~~~~~~~~~~~

Charges :math:`\rho_j` are scalars, the output :math:`u` is a scalar.

**3D** — the Coulomb kernel :math:`K = 1/r`:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \frac{\rho_j}{r}.

**2D** — the logarithmic kernel :math:`K = \log r`:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \rho_j \log r .

Yukawa (``DMK_YUKAWA``)
~~~~~~~~~~~~~~~~~~~~~~~

Screened Coulomb. Charges :math:`\rho_j` are scalars, the output :math:`u` is a scalar. The
screening parameter :math:`\lambda > 0` is passed as ``fparam``.

**3D** — :math:`K = e^{-\lambda r}/r`:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \rho_j \, \frac{e^{-\lambda r}}{r}.

**2D** — :math:`K = K_0(\lambda r)`, the modified Bessel function of the second kind of order
zero:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \rho_j \, K_0(\lambda r).

Sqrt-Laplace (``DMK_SQRT_LAPLACE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Green's function of :math:`\sqrt{-\Delta}` up to normalization, i.e. :math:`K = r^{1-d}`.
Charges :math:`\rho_j` are scalars, the output :math:`u` is a scalar. Note that in 3D this is
:math:`1/r^2`, not :math:`1/\sqrt{r}`.

**3D** — :math:`K = 1/r^2`:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \frac{\rho_j}{r^2}.

**2D** — :math:`K = 1/r`:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \frac{\rho_j}{r}.

Laplace dipole (``DMK_LAPLACE_DIPOLE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The derivative of the Laplace kernel along a per-source dipole vector, :math:`K = d_j \cdot
\nabla_{y} K_{\mathrm{Laplace}}`. Here the charge for source :math:`j` is the vector
:math:`d_j` (``n_dim`` components per source); the output :math:`u` is a scalar.

**3D**:

.. math::

   u(x_i) = \sum_{j=1}^{N_S} \frac{d_j \cdot R}{r^3}

**2D**:

.. math::

   u(x_i) = -\sum_{j=1}^{N_S} \frac{d_j \cdot R}{r^2}

Stokeslet (``DMK_STOKESLET``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Green's function of incompressible Stokes flow. The charge for source :math:`j` is a force
vector :math:`f_j` (3 components); the output :math:`u` is a velocity (3 components).
Three-dimensional only.

**3D** — :math:`G_{kl} = \tfrac{1}{2}\big(\delta_{kl}/r + R_k R_l / r^3\big)`:

.. math::

   u_k(x_i) = \frac{1}{2} \sum_{j=1}^{N_S}
              \left( \frac{f_{j,k}}{r} + \frac{(f_j \cdot R)\, R_k}{r^3} \right),
   \qquad k = 1,2,3.

Stresslet (``DMK_STRESSLET``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Stokes double-layer kernel. The charge for source :math:`j` is a density vector
:math:`\mu_j` (3 components) and the orientation :math:`\nu_j` is supplied separately through
``normal`` (3 components); the output :math:`u` is a velocity (3 components).
Three-dimensional only.

**3D**:

.. math::

   u_k(x_i) = -3 \sum_{j=1}^{N_S}
              \frac{(\mu_j \cdot R)\, (\nu_j \cdot R)\, R_k}{r^5},
   \qquad k = 1,2,3.

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
   * - ``DMK_VELOCITY``
     - Velocity field (Stokes kernels).
