DMK: Dual-space Multilevel Kernel-splitting
===========================================

DMK is a C++ framework for fast evaluation of radially symmetric, non-oscillatory
kernels. It is a dimension- and kernel-independent fast algorithm for computing discrete
summations

.. math::

   u(x_i) = \sum_{j=1}^{N_S} K(x_i, y_j)\, \rho_j, \quad i = 1, \ldots, N_T

or the continuous analog

.. math::

   u(x) = \int_{B} K(x, y)\, \rho(y)\, dy.

Here the kernel :math:`K` is radially symmetric, :math:`K(x, y) = K(|x - y|)`, and
non-oscillatory. This class covers a wide range of kernels in mathematical physics,
statistics, and machine learning, including the Green's functions of the Poisson, Yukawa,
and incompressible Stokes equations, the power functions :math:`1/r^\beta`, and the radial
basis and Matérn kernels common in statistics and machine learning.

The framework uses a hierarchy of grids: a smoothed interaction is computed at the coarsest
level, followed by corrections at finer and finer scales until the problem is entirely
local, at which point direct summation is applied. The interaction at each scale is
diagonalized by a short Fourier transform, unifying the fast multipole method, Ewald
summation, and multilevel summation while achieving speeds comparable to the FFT in work per
gridpoint, even in a fully adaptive context.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install
   math
   quickstart
   api
   refs
