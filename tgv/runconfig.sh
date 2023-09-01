#!/bin/bash
#

CXX="mpicxx"

cmake -B build -DCMAKE_CXX_COMPILER=${CXX}

cmake --build ./build
