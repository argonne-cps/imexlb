#!/bin/bash

mpirun -hostfile $COBALT_NODEFILE \
	 -np 128 \
	 -npernode 128 \
	 -cpus-per-proc 1 \
	 ./LBM.exe
