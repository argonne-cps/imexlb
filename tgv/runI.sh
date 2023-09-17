#!/bin/bash -l
CASE=tgv256
BACKEND=cpu

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=4 # Number of MPI ranks to spawn per node
NDEPTH=2 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$((NNODES*NRANKS))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

#run
mpiexec --np ${NTOTRANKS} \
	-ppn ${NRANKS} \
	-d ${NDEPTH} \
	--cpu-bind depth \
	-env OMP_NUM_THREADS=${NTHREADS} \
	./LBM.exe | tee ${CASE}-${BACKEND}-run-${NTOTRANKS}.log
