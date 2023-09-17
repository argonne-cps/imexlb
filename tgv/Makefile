#KOKKOS_PATH=/grand/projects/IMEXLBM/czhao/Kokkos/kokkos
#KOKKOS_DEVICES="Cuda"
#KOKKOS_DEVICES="Serial"
EXE_NAME="LBM"
MKDIR=mkdir

#SRC=$(wildcard *.cpp)
#SRC+=$(wildcard ./src/*.cpp)

SRC1=$(wildcard *.cpp)
SRC2=$(wildcard ./src/*.cpp)

default: build
	echo "Start Build"

CXX = CC
EXE = ${EXE_NAME}.exe
#KOKKOS_ARCH = "Zen"
#KOKKOS_ARCH = "Ampere80"
#KOKKOS_CUDA_OPTIONS = "enable_lambda"

CXXFLAGS = -O3 -g
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cpp=.o)
SRC=./
OBJ=*.o
LIB =

##include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ1) $(OBJ2) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: 
	rm -rf *.o *.cuda *.host *.dat *.exe *.plt

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) -I./src $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
