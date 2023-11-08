#include "mpi.h"
#include "lbm.hpp"
#include "System.hpp"
#include <Kokkos_Core.hpp>

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
        if (rank == 0)
        {
            s1.Monitor();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        LBM l1(MPI_COMM_WORLD, s1.sx, s1.sy, s1.sz, s1.tau, s1.rho0, s1.u0);

        l1.Initialize();
        l1.setup_subdomain();
 
        //warm-up	
        for (int it = 1; it <= 20; it++)
        {
            l1.Collision();
            l1.pack();
            l1.exchange();
            l1.unpack();
            l1.Streaming();
            l1.Update();
            if (l1.comm.me == 0) {
        	std::cout << "Warm-Up, Time-Step: " << it << std::endl; 
    	    }

	}
     
        if (l1.comm.me == 0) {
        	std::cout << "Reset and Run Main-Loop" << std::endl; 
        }        

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

            //Update
	    start_Update = MPI_Wtime();
            {
	       l1.Update();
            }
	    end_Update = MPI_Wtime();
            time_Update += end_Update - start_Update;

            if (l1.comm.me == 0) printf("time-step = %f\n", (double) it);

        }
        end = MPI_Wtime();

    double time_Total = end - start;

    double avgTime;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Col, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg time spent in Collision: " << avgTime << std::endl;
    }
 
    avgTime=0.0;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Pack, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg time spent in Pack: " << avgTime << std::endl;
    }
    
    avgTime=0.0;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Exchange, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg time spent in Exchange: " << avgTime << std::endl;
    }
    
    avgTime=0.0;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Unpack, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg time spent in Unpack: " << avgTime << std::endl;
    }
    
    avgTime=0.0;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Stream, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg time spent in Stream: " << avgTime << std::endl;
    }
    
    avgTime=0.0;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Update, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg time spent in Update: " << avgTime << std::endl;
    }
    
    avgTime=0.0;
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Reduce(&time_Total, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (l1.comm.me == 0) {
        avgTime /= nranks;
        std::cout << "Avg Total Solver Time: " << avgTime << std::endl;
        std::cout << "Avg Time per time-step: " << avgTime/s1.Time << std::endl;
    }

    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
