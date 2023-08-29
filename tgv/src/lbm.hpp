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
#include <mpi.h>

#define q 27
#define dim 3
#define ghost 3

struct CommHelper
{

    MPI_Comm comm;
    // rank number for each dim
    int rx, ry, rz;
    // rank
    int me;
    // axis for each rank
    int px, py, pz;
    int nranks;
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
        left  = px == 0      ? me+rx-1         : me - 1;
        right = px == rx - 1 ? me-rx+1         : me + 1;

        down  = pz == 0      ? me+rx*(rz-1)    : me - rx;
        up    = pz == rz - 1 ? me-rx*(rz-1)    : me + rx;

        front = py == 0      ? me+rx*rz*(ry-1) : me - rx * rz;
        back  = py == ry - 1 ? me-rx*rz*(ry-1) : me + rx * rz;

        // 12 edges
        leftup    = (pz == rz - 1) ? left-rx*(rz-1)  : left + rx;
        rightup   = (pz == rz - 1) ? right-rx*(rz-1) : right + rx;
        leftdown  = (pz == 0)      ? left+rx*(rz-1)  : left - rx;
        rightdown = (pz == 0)      ? right+rx*(rz-1) : right - rx;

        frontup    = (pz == rz - 1) ? front-rx*(rz-1)  : front + rx;
        frontright = (px == rx - 1) ? front -rx+1      : front + 1;
        frontdown  = (pz == 0)      ? front +rx*(rz-1) : front - rx;
        frontleft  = (px == 0)      ? front +rx-1      : front - 1;

        backup    = (pz == rz - 1) ? back -rx*(rz-1) : back + rx;
        backright = (px == rx - 1) ? back -rx+1      : back + 1;
        backdown  = (pz == 0)      ? back +rx*(rz-1) : back - rx;
        backleft  = (px == 0)      ? back +rx-1      : back - 1;

        // 8 points
        backrightdown  = (px == rx - 1 )  ? backdown-rx+1 : backdown + 1;
        frontrightdown = (px == rx - 1 )  ? frontdown-rx+1 : frontdown + 1;
        frontrightup   = (px == rx - 1 )  ? frontup-rx+1   : frontup + 1;
        backrightup    = (px == rx - 1 )  ? backup-rx+1    : backup + 1;
        frontleftup    = (px == 0 )       ? frontup+rx-1   : frontup - 1;
        backleftdown   = (px == 0 )       ? backdown+rx-1  : backdown - 1;
        frontleftdown  = (px == 0 )       ? frontdown+rx-1 : frontdown - 1;
        backleftup     = (px == 0 )       ? backup+rx-1    : backup - 1;

        printf("Me:%i MyNeibors: %i %i \n", me, frontleftup, backrightdown);

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

    int ex, ey, ez;

    // local axis
    int x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0, z_lo = 0, z_hi = 0;
    double rho0, mu, cs2, tau0, u0;

    // 6 faces
    double *test,*test1;
    double *m_left, *m_right, *m_down, *m_up, *m_front, *m_back;
    double *m_leftout, *m_rightout, *m_downout, *m_upout, *m_frontout, *m_backout;

    // 12 edges
    double *m_leftup, *m_rightup, *m_leftdown, *m_rightdown, *m_frontup, *m_backup, *m_frontdown, *m_backdown, *m_frontleft, *m_backleft, *m_frontright, *m_backright;
    double *m_leftupout, *m_rightupout, *m_leftdownout, *m_rightdownout, *m_frontupout, *m_backupout, *m_frontdownout, *m_backdownout, *m_frontleftout, *m_backleftout, *m_frontrightout, *m_backrightout;
  
    // 8 points
    double *m_frontleftup, *m_frontrightup, *m_frontleftdown, *m_frontrightdown, *m_backleftup, *m_backleftdown, *m_backrightup, *m_backrightdown;
    double *m_frontleftupout, *m_frontrightupout, *m_frontleftdownout, *m_frontrightdownout, *m_backleftupout, *m_backleftdownout, *m_backrightupout, *m_backrightdownout;
    MPI_Datatype m_face[3], m_line[3], m_point;

    // particle distribution eqution
    // bounce back notation
    // weight function
    // discrete velocity
    double *f, *ft, *fb, *ua, *va, *wa, *rho, *p, *t;
    int *e, *usr, *ran, *bb;

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

        ex = l_l[0];
        ey = l_l[1];
        ez = l_l[2];


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

        f = (double *)malloc(sizeof(double) * q * lx * ly * lz);
        ft = (double *)malloc(sizeof(double) * q * lx * ly * lz);
        fb = (double *)malloc(sizeof(double) * q * lx * ly * lz);

        ua = (double *)malloc(sizeof(double) * lx * ly * lz);
        va = (double *)malloc(sizeof(double) * lx * ly * lz);
        wa = (double *)malloc(sizeof(double) * lx * ly * lz);

        rho = (double *)malloc(sizeof(double) * lx * ly * lz);
        p = (double *)malloc(sizeof(double) * lx * ly * lz);

        e = (int *)malloc(sizeof(int) * q * dim);
        t = (double *)malloc(sizeof(double) * q);
        usr = (int *)malloc(sizeof(int) * lx * ly * lz);
        ran = (int *)malloc(sizeof(int) * lx * ly * lz);
        bb = (int *)malloc(sizeof(int) * q);

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Type_vector(1, q * lx * ly, 1, MPI_DOUBLE, &m_face[0]);
        MPI_Type_commit(&m_face[0]);

        MPI_Type_vector(lz, q * lx, q * lx * ly, MPI_DOUBLE, &m_face[1]);
        MPI_Type_commit(&m_face[1]);

        MPI_Type_vector(ly * lz, q, lx * q, MPI_DOUBLE, &m_face[2]);
        MPI_Type_commit(&m_face[2]);
        // line
        MPI_Type_vector(1, q * lx, 1, MPI_DOUBLE, &m_line[0]);
        MPI_Type_commit(&m_line[0]);

        MPI_Type_vector(ly, q, q * lx, MPI_DOUBLE, &m_line[1]);
        MPI_Type_commit(&m_line[1]);

        MPI_Type_vector(lz, q, lx * ly * q, MPI_DOUBLE, &m_line[2]);
        MPI_Type_commit(&m_line[2]);
        // point
        MPI_Type_vector(1, q, 1, MPI_DOUBLE, &m_point);
        MPI_Type_commit(&m_point);

        MPI_Barrier(MPI_COMM_WORLD);        

        //printf("Me is %d, x_lo=%d,x_hi=%d\n", comm.me,x_lo, x_hi);
    
        //Total Number of Points and Total number of points per process
        int myTotalPts = l_l[0]*l_l[1]*l_l[2];
        for (int i = 0; i < comm.nranks; i++)
	{
           if (i == comm.me) 
	   {
            printf("Rank %d has %d points\n", comm.me, myTotalPts);
	   }
        }
	MPI_Barrier(MPI_COMM_WORLD);

    };

    void Initialize();
    void Collision();
    //void setup_subdomain();
    //void pack();
    void exchange();
    //void unpack();
    void Streaming();
    void Boundary();
    void Update();
    //void Update1();
    void MPIoutput(int n);
    void Output(int n);
   
};
#endif
