#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
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
    int rx, ry, rz;
    int nranks;
    int me;
    int px, py, pz;
    int up, down, left, right, front, back, frontup, frontdown, frontleft, frontright, frontleftup, frontleftdown, frontrightup, frontrightdown, backup, backdown, backleft, backright, backleftup, backrightup, backleftdown, backrightdown, leftup, leftdown, rightup, rightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        // int nranks;
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = std::pow(1.0 * nranks, 1.0 / 3.0);
        while (nranks % rx != 0)
            rx++;

        rz = std::sqrt(1.0 * (nranks / rx));
        while ((nranks / rx) % rz != 0)
            rz++;

        ry = nranks / rx / rz;

        // printf("rx=%i,ry=%i,rz=%i\n", rx, ry, rz);
        px = me % rx;
        pz = (me / rx) % rz;
        py = (me / rx / rz);

        left = px == 0 ? -1 : me - 1;
        leftup = (px == 0 || pz == rz - 1) ? -1 : me - 1 + rx;
        rightup = (px == rx - 1 || pz == rz - 1) ? -1 : me + 1 + rx;
        leftdown = (px == 0 || pz == 0) ? -1 : me - 1 - rx;
        rightdown = (px == rx - 1 || pz == 0) ? -1 : me + 1 - rx;
        right = px == rx - 1 ? -1 : me + 1;
        down = pz == 0 ? -1 : me - rx;
        up = pz == rz - 1 ? -1 : me + rx;

        front = py == 0 ? -1 : me - rx * rz;
        frontup = (py == 0 || pz == rz - 1) ? -1 : me - rx * rz + rx;
        frontdown = (py == 0 || pz == 0) ? -1 : me - rx * rz - rx;
        frontleft = (py == 0 || px == 0) ? -1 : me - rx * rz - 1;
        frontright = (py == 0 || px == rx - 1) ? -1 : me - rx * rz + 1;
        frontleftdown = (py == 0 || px == 0 || pz == 0) ? -1 : me - rx * rz - rx - 1;
        frontrightdown = (py == 0 || px == rx - 1 || pz == 0) ? -1 : me - rx * rz - rx + 1;
        frontrightup = (py == 0 || px == rx - 1 || pz == rz - 1) ? -1 : me - rx * rz + rx + 1;
        frontleftup = (py == 0 || px == 0 || pz == rz - 1) ? -1 : me - rx * rz + rx - 1;

        back = py == ry - 1 ? -1 : me + rx * rz;
        backup = (py == ry - 1 || pz == rz - 1) ? -1 : me + rx * rz + rx;
        backdown = (py == ry - 1 || pz == 0) ? -1 : me + rx * rz - rx;
        backleft = (py == ry - 1 || px == 0) ? -1 : me + rx * rz - 1;
        backright = (py == ry - 1 || px == rx - 1) ? -1 : me + rx * rz + 1;
        backleftdown = (py == ry - 1 || px == 0 || pz == 0) ? -1 : me + rx * rz - rx - 1;
        backrightdown = (py == ry - 1 || px == rx - 1 || pz == 0) ? -1 : me + rx * rz - rx + 1;
        backrightup = (py == ry - 1 || px == rx - 1 || pz == rz - 1) ? -1 : me + rx * rz + rx + 1;
        backleftup = (py == ry - 1 || px == 0 || pz == rz - 1) ? -1 : me + rx * rz + rx - 1;

        MPI_Barrier(MPI_COMM_WORLD);
    }
    template <class ViewType>
    void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request *request_send, MPI_Request *request_recv)
    {
        MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
        MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
    }
};
struct LBM
{

    CommHelper comm;

    int mpi_active_requests;

    int glx;
    int gly;
    int glz;

    // exact nodes
    // 256/1,64/2,64/1
    int ex;
    int ey;
    int ez;
    // include ghost nodes
    // 256+2*3,32+2*3,64+2*3
    int lx;
    int ly;
    int lz;
    // 3,3,3  262,35,67
    // 256,32,64
    int l_s[3];
    int l_e[3];
    int l_l[3];

    int x_lo = 0;
    int x_hi = 0;
    int y_lo = 0;
    int y_hi = 0;
    int z_lo = 0;
    int z_hi = 0;
    double rho0;
    double mu;
    double cs2;
    double tau0;
    double u0;

    double *m_left, *m_right, *m_down, *m_up, *m_front, *m_back;
    double *m_leftout, *m_rightout, *m_downout, *m_upout, *m_frontout, *m_backout;
    // 12 edges
    double *m_leftup, *m_rightup, *m_leftdown, *m_rightdown, *m_frontup, *m_backup, *m_frontdown, *m_backdown, *m_frontleft, *m_backleft, *m_frontright, *m_backright;
    double *m_leftupout, *m_rightupout, *m_leftdownout, *m_rightdownout, *m_frontupout, *m_backupout, *m_frontdownout, *m_backdownout, *m_frontleftout, *m_backleftout, *m_frontrightout, *m_backrightout;
    // 8 points
    double *m_frontleftup, *m_frontrightup, *m_frontleftdown, *m_frontrightdown, *m_backleftup, *m_backleftdown, *m_backrightup, *m_backrightdown;
    double *m_frontleftupout, *m_frontrightupout, *m_frontleftdownout, *m_frontrightdownout, *m_backleftupout, *m_backleftdownout, *m_backrightupout, *m_backrightdownout;
    MPI_Datatype m_face[3], m_line[3], m_point;
    double *f, *ft, *fb, *ua, *va, *wa, *rho, *p, *t;
    int *e, *usr, *ran, *bb;

    LBM(MPI_Comm comm_, int sx, int sy, int sz, double &tau, double &rho0, double &u0) : comm(comm_), glx(sx), gly(sy), glz(sz), tau0(tau), rho0(rho0), u0(u0)
    {

        l_l[0] = (comm.px - glx % comm.rx >= 0) ? glx / comm.rx : glx / comm.rx + 1;
        l_l[1] = (comm.py - gly % comm.ry >= 0) ? gly / comm.ry : gly / comm.ry + 1;
        l_l[2] = (comm.pz - glz % comm.rz >= 0) ? glz / comm.rz : glz / comm.rz + 1;
        // local length
        lx = l_l[0] + 2 * ghost;
        ly = l_l[1] + 2 * ghost;
        lz = l_l[2] + 2 * ghost;

        ex = l_l[0];
        ey = l_l[1];
        ez = l_l[2];

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
    };

    void Initialize();
    void Collision();
    void exchange();
    void Streaming();
    void Update();
    void MPIoutput(int n);
    void Output(int n);
};
#endif
