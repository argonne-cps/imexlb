# Lid Driven Cavity

## Background
The lid-driven-cavity problem is one of the most important benchmarks for computational fluid dynamic (CFD) solvers. 
Its importance results from the fundamental rectangular or square geometry and the simple driving of the flow by means of the tangential motion with constant velocity of a single lid, representing Dirichlet boundary conditions. Moreover, the driven-cavity flow exhibits a number of interesting physical features. 

## Code Characteristics 
* Written in C/C++.
* MPI+X hybrid parallelism model where X utilizes the Kokkos (with CUDA backend) programming model.  
* Written for 3D problems where the D3Q27 model is used to discretize the velocity-space in the Boltzmann equation. 

## Building
