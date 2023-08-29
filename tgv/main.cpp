#include "mpi.h"
#include <Kokkos_Core.hpp>
#include "lbm.hpp"
#include "System.hpp"

int main(int argc, char *argv[])
{

  double start, end;
  double start_Col, end_Col;
  double start_Pack, end_Pack;
  double start_Exchange, end_Exchange;
  double start_Unpack, end_Unpack;
  double start_Stream, end_Stream;
  double start_Update, end_Update;

  double time_Col, time_Pack, time_Exchange;
  double time_Unpack, time_Stream, time_Update;      

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {

        int rank;
        int nranks;
	      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);

        System s1(rank); 
        {
            s1.Monitor();
        }        
        MPI_Barrier(MPI_COMM_WORLD);
        LBM l1(MPI_COMM_WORLD, s1.sx, s1.sy, s1.sz, s1.tau, s1.rho0, s1.u0);

        l1.Initialize();
        //l1.MPIoutput(0);
        l1.setup_subdomain();

        start = MPI_Wtime();
        for (int it = 1; it <= s1.Time; it++)
        {

           	//collision
            start_Col = MPI_Wtime();
	          {
	            l1.Collision();
	          }
	          end_Col = MPI_Wtime();
	          time_Col += end_Col - start_Col;

            //pack
            start_Pack = MPI_Wtime();
	          {
	            l1.pack();
            }
	          end_Pack = MPI_Wtime();
	          time_Pack += end_Pack - start_Pack;

            //exchange
            start_Exchange = MPI_Wtime();
            {
      	       l1.exchange();
            }
	          end_Exchange = MPI_Wtime();
	          time_Exchange += end_Exchange - start_Exchange;

            //unpack
            start_Unpack = MPI_Wtime();
	          {
	            l1.unpack();
            }
            end_Unpack = MPI_Wtime();
	          time_Unpack += end_Unpack - start_Unpack;
            
            //Stream
            start_Stream = MPI_Wtime();
      	    {
	            l1.Streaming();
            }
	          end_Stream = MPI_Wtime();
            time_Stream += end_Stream - start_Stream;
     
	          if (it % s1.inter == 0)
            {
                l1.MPIoutput(it / s1.inter);
            }
            if (l1.comm.me == 0) printf("time-step = %f\n", (double) it);
        }
        end = MPI_Wtime();
        double time_Total = end - start; 

  }       

  Kokkos::finalize();
  MPI_Finalize();

  return 0;

}
