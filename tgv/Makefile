#KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_PATH=/gpfs/jlse-fs0/users/spatel/kokkos/amd_MI100/kokkos/
#KOKKOS_DEVICES = "OpenMP"
KOKKOS_DEVICES = "HIP"
EXE_NAME = "lbm_tgv"

#SRC = $(wildcard *.cpp)
SRC1=$(wildcard *.cpp)
SRC2=$(wildcard ./src/*.cpp)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
#CXX = g++
#CXX = hipcc
CXX = mpicxx
EXE = ${EXE_NAME}.hip
KOKKOS_ARCH = "Vega908"
endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

#OBJ = $(SRC:.cpp=.o)
OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cpp=.o)
SRC=./
OBJ=*.o
LIB =
include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ1) $(OBJ2) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) -I./src $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
