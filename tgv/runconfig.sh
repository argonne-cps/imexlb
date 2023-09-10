#!/bin/bash
#

CXX="CC"
cmake -B build -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=cc -DKokkos_ROOT=${KOKKOS_INSTALL}/lib64/cmake/Kokkos
cmake --build ./build
