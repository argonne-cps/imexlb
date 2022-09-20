# imexlb
A repo for performance-portable, lattice Boltzmann solvers on heterogeneous platforms 

## Purpose
The IMEXLB project aims to develop a Lattice Boltzmann Method (LBM) proxy application code-suite for heterogeneous platforms (such as ThetaGPU). A ProxyApp, by definition, is a proxy for a full-fledged application code that simulates a wider array of problems. The IMEXLB ProxyApp, therefore, is a small self-contained code unit, with minimal dependencies, that will demonstrate both the science as well as the route to achieving parallel performance on a heterogeneous computing platform for one or more exemplar problems.

For this project, in particular, the IMEXLB ProxyApp suite will consist of Fortran and C++ codes that will simulate single and two-phase canonical flows using the MPI+X parallelization paradigm, where X denotes the SIMD parallelization paradigms: OpenMP-4.5+, Kokkos, DPC++. This ProxyApp will serve as a reference for other LB code developers who may wish to either use IMEXLB as a building block, or follow the SIMD parallelization strategy to modify their own code, to do their simulations on platforms like Polaris, Aurora, & Frontier.

## Background
LBM is a relatively novel approach to solve the Navier-Stokes equations (NSE) in the low-Mach number regime. The governing equation can be derived from the Boltzmann equation after discretizing the phase space with constant microscopic lattice velocities. One major drive behind the use of LBM in the CFD community is the ease of parallelization, but the increasing popularity of LBM can also be attributed to its unique feature: LBM solves a simplified Boltzmann equation that is essentially a set of 1st-order hyperbolic PDEs with constant microscopic lattice velocities, for which a plethora of simple yet excellent discretization schemes are available. Furthermore, all the complex non-linear effects are modeled locally at the grid points.

## Code Characteristics 
* Written in Fortran 90 and C/C++. 
* MPI+X hybrid parallelism with OpenMP-4.5+, SYCL/DPC++, and Kokkos (with CUDA backend) programming models  
* 2D (D2Q9) and 3D (D3Q27) problems 
* Example problems include: flow past a circle/sphere, lid driven cavity, taylor-green vortex

## Building
The code has currently been tested on ALCF's [ThetaGPU](https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview), as such, the following instructions assume you are building on ThetaGPU. In the future, we will provide more general instructions for non-ThetaGPU architectures.  
