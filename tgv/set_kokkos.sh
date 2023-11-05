#!/bin/bash

#This is for running Kokkos_SYCL on Polaris
module load oneapi
module load cudatoolkit-standalone/11.8.0
module load cmake 

export ROOT_DIR="/lus/grand/projects/IMEXLBM/spatel/kokkos"
export KOKKOS_INSTALL=${ROOT_DIR}/polaris_sycl/install/kokkos
export KOKKOS_HOME=${KOKKOS_INSTALL}

echo "-------------------------------"
echo "KOKKOS_HOME="${KOKKOS_HOME}
echo "-------------------------------"  

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${KOKKOS_HOME}/lib64
export CPATH=$CPATH:${KOKKOS_HOME}/include
export LIBRARY_PATH=$LIBRARY_PATH:${KOKKOS_HOME}/lib64
