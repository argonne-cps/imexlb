# Taylor-Green Vortex

## Background
The Taylor-Green vortex (TGV) is a benchmark flow case that is often used to study the accuracy of capturing turbulent phenomena and performance of a CFD code. The initial condition is smooth and three dimensional.  The simulation evolves with the generation of small-scale vorticity and vortical structures.  These structures transition the flow into turbulence.           

## Code Characteristics 

## Building & Running

On ThetaGPU: 

1. Please modify the `KOKKOS_HOME` variable in the makefile. This should point to your local build of Kokkos. 

2. do 

```
export OMPI_MPICXX=${KOKKOS_HOME}/bin/nvcc_wrapper  
```

On Polaris

