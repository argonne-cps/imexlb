#include "mpi.h"
#include "lbm.hpp"
#include "System.hpp"

int main(int argc, char *argv[])
{

    double start, end;
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    System s1;
    if (rank == 0)
    {
        s1.Monitor();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    LBM l1(MPI_COMM_WORLD, s1.sx, s1.sy, s1.sz, s1.tau, s1.rho0, s1.u0);

    l1.Initialize();

   l1.MPIoutput(0);
    start = MPI_Wtime();
    for (int it = 1; it <= s1.Time; it++)
    {
        l1.Collision();
        l1.exchange();
        l1.Streaming();
        l1.Update();
        end = MPI_Wtime();
        if (it % s1.inter == 0)
        {
            l1.MPIoutput(it / s1.inter);
            if (l1.comm.me == 0)
                printf("time=%f\n", end - start);
        }
	}

    MPI_Finalize();

    return 0;
}
