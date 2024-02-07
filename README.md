# DMK
A Dual-space Multilevel Kernel-splitting (DMK) Framework for Discrete and Continuous Convolution

# Introduction

The DMK framework is a dimension-independent and kernel-independent fast algorithm for computing
discrete summations

$$u(x_i) = \sum_{j=1}^{N_S} K(x_i, y_j) \rho_j, \text{for~} i = 1, \ldots, N_T$$

or the continuous analog

$$u(x) = \int_{B} K(x,y) \rho(y) dy.$$

Here the kernel $K$ is assumed to be radially symmetric, i.e., $K(x,y)=K(|x-y|)$ and
nonoscillatory. This class of kernels cover a wide range of kernels in mathematical physics,
statistics, and machine learning. For example, the Green's functions of classic PDEs such as the
Poisson equation, the Yukawa equation, the incompressible Stokes equations all belong to the
class. The kernel of fractional PDEs, i.e., the power function $1/r^\beta$ for arbitrary nonnegative
real number $\beta$ also belongs to this class. Many radial basis functions used in the kernel method
in statistics clearly belong to the class. And the so-called Matern kernels that are widely used in
statistics and machine learning also belong to this class.

The DMK (dual-space multilevel kernel-splitting) framework uses a hierarchy of grids, computing a
smoothed interaction at the coarsest level, followed by a sequence of corrections at finer and finer
scales until the problem is entirely local, at which point direct summation is applied. The main
novelty of DMK is that the interaction at each scale is diagonalized by a short Fourier transform,
permitting the use of separation of variables, but without requiring the FFT for its asymptotic
performance. The DMK framework substantially simplifies the algorithmic structure of the fast
multipole method (FMM) and unifies the FMM, Ewald summation, and multilevel summation, achieving
speeds comparable to the FFT in work per gridpoint, even in a fully adaptive context. For continuous
source distributions, the evaluation of local interactions is further accelerated by approximating
the kernel at the finest level as a sum of Gaussians with a highly localized remainder. The Gaussian
convolutions are calculated using tensor product transforms, and the remainder term is calculated
using asymptotic methods.

The tree is a level-restricted (i.e., 2:1 balanced) adaptive tree.

# Installation Guide

We use make utility to install static and/or dynamic libraries, and to run the tests. 
Type "make" in the main directory to see the list of options for compiling the point code.
The box code is compiled and tested using the makefile in test/bdmk.

The box code uses BLAS and we suggest that the user use the Intel compiler ifort and the 
Intel MKL library for optimal performance. Please do "ulimit -s unlimited" on the command 
window to avoid segfault before carrying out high-accuracy calculations. The point code uses 
SCTL from PVFMM by Dhairya Malhotra and VCL by Agner Fog for SIMD accelerated kernel evaluations. 

# Main subroutines

1. The point code is src/pdmk/pdmk.f

2. The box code is src/bdmk/bdmk.f, which requires calling subroutines
vol_tree_mem and vol_tree_build in src/common/tree_vol_coeffs.f first to build the tree.

