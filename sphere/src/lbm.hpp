#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <mpi.h>

#define q 27
#define dim 3
#define ghost 3

typedef Kokkos::RangePolicy<> range_policy;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<4>> mdrange_policy4;

using buffer_t = Kokkos::View<double ***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using buffer_ut = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using buffer_st = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

struct CommHelper
{

    MPI_Comm comm;
    // rank number for each dim
    int rx, ry, rz;
    // rank
    int me;
    int nranks;
    // axis for each rank
    int px, py, pz;

    // 6 faces
    int up, down, left, right, front, back;
    // 12 edges
    int frontup, frontdown, frontleft, frontright, backup, backdown, backleft, backright, leftup, leftdown, rightup, rightdown;
    // 8 points
    int frontleftup, frontleftdown, frontrightup, frontrightdown, backleftup, backrightup, backleftdown, backrightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;

        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = std::pow(1.0 * nranks, 1.0 / 3.0);
        while (nranks % rx != 0)
            rx++;

        rz = std::sqrt(1.0 * (nranks / rx));
        while ((nranks / rx) % rz != 0)
            rz++;

        ry = nranks / rx / rz;

        px = me % rx;
        pz = (me / rx) % rz;
        py = (me / rx / rz);
        // 6 faces
        left = px == 0 ? -1 : me - 1;
        right = px == rx - 1 ? -1 : me + 1;
        down = pz == 0 ? -1 : me - rx;
        up = pz == rz - 1 ? -1 : me + rx;
        front = py == 0 ? -1 : me - rx * rz;
        back = py == ry - 1 ? -1 : me + rx * rz;
        // 12 edges
        leftup = (px == 0 || pz == rz - 1) ? -1 : me - 1 + rx;
        rightup = (px == rx - 1 || pz == rz - 1) ? -1 : me + 1 + rx;
        leftdown = (px == 0 || pz == 0) ? -1 : me - 1 - rx;
        rightdown = (px == rx - 1 || pz == 0) ? -1 : me + 1 - rx;
        frontup = (py == 0 || pz == rz - 1) ? -1 : me - rx * rz + rx;
        frontdown = (py == 0 || pz == 0) ? -1 : me - rx * rz - rx;
        frontleft = (py == 0 || px == 0) ? -1 : me - rx * rz - 1;
        frontright = (py == 0 || px == rx - 1) ? -1 : me - rx * rz + 1;
        backup = (py == ry - 1 || pz == rz - 1) ? -1 : me + rx * rz + rx;
        backdown = (py == ry - 1 || pz == 0) ? -1 : me + rx * rz - rx;
        backleft = (py == ry - 1 || px == 0) ? -1 : me + rx * rz - 1;
        backright = (py == ry - 1 || px == rx - 1) ? -1 : me + rx * rz + 1;
        // 8 points
        frontleftdown = (py == 0 || px == 0 || pz == 0) ? -1 : me - rx * rz - rx - 1;
        frontrightdown = (py == 0 || px == rx - 1 || pz == 0) ? -1 : me - rx * rz - rx + 1;
        frontrightup = (py == 0 || px == rx - 1 || pz == rz - 1) ? -1 : me - rx * rz + rx + 1;
        frontleftup = (py == 0 || px == 0 || pz == rz - 1) ? -1 : me - rx * rz + rx - 1;
        backleftdown = (py == ry - 1 || px == 0 || pz == 0) ? -1 : me + rx * rz - rx - 1;
        backrightdown = (py == ry - 1 || px == rx - 1 || pz == 0) ? -1 : me + rx * rz - rx + 1;
        backrightup = (py == ry - 1 || px == rx - 1 || pz == rz - 1) ? -1 : me + rx * rz + rx + 1;
        backleftup = (py == ry - 1 || px == 0 || pz == rz - 1) ? -1 : me + rx * rz + rx - 1;
        // output the direction;
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

struct LBM
{

    CommHelper comm;
    MPI_Request mpi_requests_recv[26];
    MPI_Request mpi_requests_send[26];
    int mpi_active_requests;

    int glx, gly, glz;
    // include ghost nodes
    int lx, ly, lz;
    // local start, local end, local length
    int l_s[3], l_e[3], l_l[3];

    // local axis
    int x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0, z_lo = 0, z_hi = 0;
    double rho0, mu, cs2, tau0, u0;

    // 6 faces
    buffer_t m_left, m_right, m_down, m_up, m_front, m_back;
    buffer_t m_leftout, m_rightout, m_downout, m_upout, m_frontout, m_backout;
    // 12 edges
    buffer_ut m_leftup, m_rightup, m_leftdown, m_rightdown, m_frontup, m_backup, m_frontdown, m_backdown, m_frontleft, m_backleft, m_frontright, m_backright;
    buffer_ut m_leftupout, m_rightupout, m_leftdownout, m_rightdownout, m_frontupout, m_backupout, m_frontdownout, m_backdownout, m_frontleftout, m_backleftout, m_frontrightout, m_backrightout;
    // 8 points
    buffer_st m_frontleftup, m_frontrightup, m_frontleftdown, m_frontrightdown, m_backleftup, m_backleftdown, m_backrightup, m_backrightdown;
    buffer_st m_frontleftupout, m_frontrightupout, m_frontleftdownout, m_frontrightdownout, m_backleftupout, m_backleftdownout, m_backrightupout, m_backrightdownout;
    // particle distribution eqution
    Kokkos::View<double ****, Kokkos::CudaUVMSpace> f, ft, fb;
    // macro scopic equation
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> ua, va, wa, rho, p;
    // usr define
    Kokkos::View<int ***, Kokkos::CudaUVMSpace> usr, ran;
    // bounce back notation
    Kokkos::View<int *, Kokkos::CudaUVMSpace> bb;
    // weight function
    Kokkos::View<double *, Kokkos::CudaUVMSpace> t;
    // discrete velocity
    Kokkos::View<int **, Kokkos::CudaUVMSpace> e;

    LBM(MPI_Comm comm_, int sx, int sy, int sz, double &tau, double &rho0, double &u0) : comm(comm_), glx(sx), gly(sy), glz(sz), tau0(tau), rho0(rho0), u0(u0)
    {
        // local length
        l_l[0] = (comm.px - glx % comm.rx >= 0) ? glx / comm.rx : glx / comm.rx + 1;
        l_l[1] = (comm.py - gly % comm.ry >= 0) ? gly / comm.ry : gly / comm.ry + 1;
        l_l[2] = (comm.pz - glz % comm.rz >= 0) ? glz / comm.rz : glz / comm.rz + 1;
        // local length
        lx = l_l[0] + 2 * ghost;
        ly = l_l[1] + 2 * ghost;
        lz = l_l[2] + 2 * ghost;
        // local start
        l_s[0] = ghost;
        l_s[1] = ghost;
        l_s[2] = ghost;
        // local end
        l_e[0] = l_s[0] + l_l[0];
        l_e[1] = l_s[1] + l_l[1];
        l_e[2] = l_s[2] + l_l[2];

        int x_his[comm.nranks];
        int y_his[comm.nranks];
        int z_his[comm.nranks];
        int ax_his[comm.rx][comm.ry][comm.rz];
        int ay_his[comm.rx][comm.ry][comm.rz];
        int az_his[comm.rx][comm.ry][comm.rz];

        MPI_Allgather(l_l, 1, MPI_INT, x_his, 1, MPI_INT, comm.comm);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgather(l_l + 1, 1, MPI_INT, y_his, 1, MPI_INT, comm.comm);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgather(l_l + 2, 1, MPI_INT, z_his, 1, MPI_INT, comm.comm);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < comm.rx; i++)
        {
            for (int j = 0; j < comm.ry; j++)
            {
                for (int k = 0; k < comm.rz; k++)
                {
                    ax_his[i][j][k] = x_his[i + j * (comm.rx) + k * (comm.rx) * (comm.ry)];
                    ay_his[i][j][k] = y_his[i + j * (comm.rx) + k * (comm.rx) * (comm.ry)];
                    az_his[i][j][k] = z_his[i + j * (comm.rx) + k * (comm.rx) * (comm.ry)];
                }
            }
        }

        for (int i = 0; i <= comm.px; i++)
        {
            x_hi += ax_his[i][0][0];
        }

        for (int j = 0; j <= comm.py; j++)
        {
            y_hi += ay_his[0][j][0];
        }

        for (int k = 0; k <= comm.pz; k++)
        {
            z_hi += az_his[0][0][k];
        }

        x_lo = x_hi - l_l[0];
        x_hi = x_hi - 1;

        y_lo = y_hi - l_l[1];
        y_hi = y_hi - 1;

        z_lo = z_hi - l_l[2];
        z_hi = z_hi - 1;

        printf("x_lo=%d,y_lo=%d,z_lo=%d,x_hi=%d,y_hi=%d,z_hi=%d\n", x_lo, y_lo, z_lo, x_hi, y_hi, z_hi);
    };

    void Initialize();
    void Collision();
    void setup_subdomain();
    void pack();
    void exchange();
    void unpack();
    void Streaming();
    void Boundary();
    void Update();
    void testout();
    void MPIoutput(int n);
    void Output(int n);
};
#endif
