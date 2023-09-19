#!/bin/bash -l
#PBS -N imexlbm_tgv
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A Catalyst

CASE=tgv256
BACKEND=gpu

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=4 # Number of MPI ranks to spawn per node
NDEPTH=8 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$((NNODES*NRANKS))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

cd /grand/IMEXLBM/spatel/imexlb/
source set_kokkos.sh

cd /grand/IMEXLBM/spatel/imexlb/tgv
export MPICH_GPU_SUPPORT_ENABLED=1

#run
mpiexec --np ${NTOTRANKS} \
	-ppn ${NRANKS} \
	-d ${NDEPTH} \
	--cpu-bind depth \
	-env OMP_NUM_THREADS=${NTHREADS} \
        ./set_affinity_gpu_polaris.sh\
	./build/lbm_tgv | tee ${CASE}-${BACKEND}-run-${NTOTRANKS}.log
