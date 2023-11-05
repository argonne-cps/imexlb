#!/bin/bash
#
CC="mpicc"
CXX="mpicxx"
PREFIX_PATHS="/opt/cray/pe/mpich/8.1.16/gtl/lib"
cmake -B build -DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_C_COMPILER=${CC} \
	-DCMAKE_PREFIX_PATH="${PREFIX_PATHS}" \
	-DKokkos_ROOT=${KOKKOS_INSTALL}/lib64/cmake/Kokkos
cmake --build ./build
