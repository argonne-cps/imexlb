cmake_minimum_required(VERSION 3.16)
project(tgv)
include(../common.cmake)
find_package(MPI REQUIRED)
set (CMAKE_CXX_STANDARD 17)

set(buildExample_SOURCE_DIR ".")

find_library(FOO_MPI_LIB mpi_gtl_cuda)
set(SRC 
	${buildExample_SOURCE_DIR}/main.cpp
	${buildExample_SOURCE_DIR}/src/lbm.cpp
	${buildExample_SOURCE_DIR}/src/System.cpp
	${buildExample_SOURCE_DIR}/src/mpi_view_transfer.cpp)

add_executable(lbm_tgv ${SRC})
target_link_libraries(lbm_tgv PRIVATE Kokkos::kokkoscore PRIVATE "${FOO_MPI_LIB}")
target_include_directories(lbm_tgv PUBLIC "${buildExample_SOURCE_DIR}/src")
