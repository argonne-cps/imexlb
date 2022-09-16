#include "lbm.hpp"

void LBM::setup_subdomain()
{
    m_front = (double *)malloc(sizeof(double) * q * lx * lz);
    m_frontout = (double *)malloc(sizeof(double) * q * lx * lz);

    m_back = (double *)malloc(sizeof(double) * q * lx * lz);
    m_backout = (double *)malloc(sizeof(double) * q * lx * lz);

    m_up = (double *)malloc(sizeof(double) * q * lx * ly);
    m_upout = (double *)malloc(sizeof(double) * q * lx * ly);

    m_down = (double *)malloc(sizeof(double) * q * lx * ly);
    m_downout = (double *)malloc(sizeof(double) * q * lx * ly);

    m_left = (double *)malloc(sizeof(double) * q * ly * lz);
    m_leftout = (double *)malloc(sizeof(double) * q * ly * lz);

    m_right = (double *)malloc(sizeof(double) * q * ly * lz);
    m_rightout = (double *)malloc(sizeof(double) * q * ly * lz);

    m_leftup = (double *)malloc(sizeof(double) * q * l_l[1]);
    m_leftupout = (double *)malloc(sizeof(double) * q * l_l[1]);

    m_rightup = (double *)malloc(sizeof(double) * q * l_l[1]);
    m_rightupout = (double *)malloc(sizeof(double) * q * l_l[1]);

    m_leftdown = (double *)malloc(sizeof(double) * q * l_l[1]);
    m_leftdownout = (double *)malloc(sizeof(double) * q * l_l[1]);

    m_rightdown = (double *)malloc(sizeof(double) * q * l_l[1]);
    m_rightdownout = (double *)malloc(sizeof(double) * q * l_l[1]);

    m_backleft = (double *)malloc(sizeof(double) * q * l_l[2]);
    m_backleftout = (double *)malloc(sizeof(double) * q * l_l[2]);

    m_backright = (double *)malloc(sizeof(double) * q * l_l[2]);
    m_backrightout = (double *)malloc(sizeof(double) * q * l_l[2]);

    m_frontleft = (double *)malloc(sizeof(double) * q * l_l[2]);
    m_frontleftout = (double *)malloc(sizeof(double) * q * l_l[2]);

    m_frontright = (double *)malloc(sizeof(double) * q * l_l[2]);
    m_frontrightout = (double *)malloc(sizeof(double) * q * l_l[2]);

    m_backdown = (double *)malloc(sizeof(double) * q * l_l[0]);
    m_backdownout = (double *)malloc(sizeof(double) * q * l_l[0]);

    m_backup = (double *)malloc(sizeof(double) * q * l_l[0]);
    m_backupout = (double *)malloc(sizeof(double) * q * l_l[0]);

    m_frontdown = (double *)malloc(sizeof(double) * q * l_l[0]);
    m_frontdownout = (double *)malloc(sizeof(double) * q * l_l[0]);

    m_frontup = (double *)malloc(sizeof(double) * q * l_l[0]);
    m_frontupout = (double *)malloc(sizeof(double) * q * l_l[0]);

    m_frontleftdown = (double *)malloc(sizeof(double) * q);
    m_frontleftdownout = (double *)malloc(sizeof(double) * q);

    m_frontrightdown = (double *)malloc(sizeof(double) * q);
    m_frontrightdownout = (double *)malloc(sizeof(double) * q);

    m_frontleftup = (double *)malloc(sizeof(double) * q);
    m_frontleftupout = (double *)malloc(sizeof(double) * q);

    m_frontrightup = (double *)malloc(sizeof(double) * q);
    m_frontrightupout = (double *)malloc(sizeof(double) * q);

    m_backleftdown = (double *)malloc(sizeof(double) * q);
    m_backleftdownout = (double *)malloc(sizeof(double) * q);

    m_backrightdown = (double *)malloc(sizeof(double) * q);
    m_backrightdownout = (double *)malloc(sizeof(double) * q);

    m_backleftup = (double *)malloc(sizeof(double) * q);
    m_backleftupout = (double *)malloc(sizeof(double) * q);

    m_backrightup = (double *)malloc(sizeof(double) * q);
    m_backrightupout = (double *)malloc(sizeof(double) * q);
}

void LBM::pack()
{
    // q faces
    if (y_hi == gly - 1)
        for (int k = 0; k < lz; k++)
        {
            for (int i = 0; i < lx; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {

                    m_frontout[ii + i * q + k * q * lx] = f(ii, i, l_s[1], k);
                }
            }
        }

    if (y_hi != gly - 1)

        for (int k = 0; k < lz; k++)
        {
            for (int i = 0; i < lx; i++)
            {

                for (int ii = 0; ii < q; ii++)
                {
                    m_backout[ii + i * q + k * q * lx] = f(ii, i, l_e[1] - 1, k);
                }
            }
        }

    if (z_hi == glz - 1)
        for (int j = 0; j < ly; j++)
        {
            for (int i = 0; i < lx; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {

                    m_downout[ii + i * q + j * q * lx] = f(ii, i, j, l_s[2]);
                }
            }
        }

    if (z_lo == 0)
        for (int j = 0; j < ly; j++)
        {
            for (int i = 0; i < lx; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {

                    m_upout[ii + i * q + j * q * lx] = f(ii, i, j, l_e[2] - 1);
                }
            }
        }

    if (x_hi == glx - 1)
        for (int k = 0; k < lz; k++)
        {
            for (int j = 0; j < ly; j++)
            {

                for (int ii = 0; ii < q; ii++)
                {

                    m_leftout[ii + j * q + k * q * ly] = f(ii, l_s[0], j, k);
                }
            }
        }

    if (x_lo == 0)
        for (int k = 0; k < lz; k++)
        {
            for (int j = 0; j < ly; j++)
            {

                for (int ii = 0; ii < q; ii++)
                {

                    m_rightout[ii + j * q + k * q * ly] = f(ii, l_e[0] - 1, j, k);
                }
            }
        }
    Kokkos::fence();
    // 12lines

    if (x_lo != 0 && z_hi != glz - 1)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_leftupout[ii + j * q] = f(ii, l_s[0], j + l_s[1], l_e[2] - 1);
            }
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_rightupout[ii + j * q] = f(ii, l_e[0] - 1, j + l_s[1], l_e[2] - 1);
            }
        }
    }

    if (x_lo != 0 && z_lo != 0)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_leftdownout[ii + j * q] = f(ii, l_s[0], j + l_s[1], l_s[2]);
            }
        }
    }

    if (x_hi != glx - 1 && z_lo != 0)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_rightdownout[ii + j * q] = f(ii, l_e[0] - 1, j + l_s[1], l_s[2]);
            }
        }
    }

    if (y_lo != 0 && z_hi != glz - 1)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_frontupout[ii + i * q] = f(ii, i + l_s[0], l_s[1], l_e[2] - 1);
            }
        }
    }

    if (y_lo != 0 && z_lo != 0)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_frontdownout[ii + i * q] = f(ii, i + l_s[0], l_s[1], l_s[2]);
            }
        }
    }

    if (y_hi != gly - 1 && z_hi != glz - 1)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_backupout[ii + i * q] = f(ii, i + l_s[0], l_e[1] - 1, l_e[2] - 1);
            }
        }
    }

    if (y_hi != gly - 1 && z_lo != 0)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_backdownout[ii + i * q] = f(ii, i + l_s[0], l_e[1] - 1, l_s[2]);
            }
        }
    }

    if (x_lo != 0 && y_lo != 0)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_frontleftout[ii + k * q] = f(ii, l_s[0], l_s[1], k + l_s[2]);
            }
        }
    }

    if (x_hi != glx - 1 && y_lo != 0)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_frontrightout[ii + k * q] = f(ii, l_e[0] - 1, l_s[1], k + l_s[2]);
            }
        }
    }

    if (x_lo != 0 && y_hi != gly - 1)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_backleftout[ii + k * q] = f(ii, l_s[0], l_e[1] - 1, k + l_s[2]);
            }
        }
    }

    if (x_hi != glx - 1 && y_hi != gly - 1)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                m_backrightout[ii + k * q] = f(ii, l_e[0] - 1, l_e[1] - 1, k + l_s[2]);
            }
        }
    }
    Kokkos::fence();

    // 8 points
    if (x_lo != 0 && z_lo != 0 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_frontleftdownout[ii] = f(ii, l_s[0], l_s[1], l_s[2]);
        }
    }

    if (x_hi != glx - 1 && z_lo != 0 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_frontrightdownout[ii] = f(ii, l_e[0] - 1, l_s[1], l_s[2]);
        }
    }

    if (x_lo != 0 && z_lo != 0 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_backleftdownout[ii] = f(ii, l_s[0], l_e[1] - 1, l_s[2]);
        }
    }

    if (x_hi != glx - 1 && z_lo != 0 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_backrightdownout[ii] = f(ii, l_e[0] - 1, l_e[1] - 1, l_s[2]);
        }
    }

    if (x_lo != 0 && z_hi != glz - 1 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_frontleftupout[ii] = f(ii, l_s[0], l_s[1], l_e[2] - 1);
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_frontrightupout[ii] = f(ii, l_e[0] - 1, l_s[1], l_e[2] - 1);
        }
    }

    if (x_lo != 0 && z_hi != glz - 1 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_backleftupout[ii] = f(ii, l_s[0], l_e[1] - 1, l_e[2] - 1);
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            m_backrightupout[ii] = f(ii, l_e[0] - 1, l_e[1] - 1, l_e[2] - 1);
        }
    }
    Kokkos::fence();
}

void LBM::exchange()
{
    // 6 faces
    int mar = 1;
    MPI_Request ttttt, ttttt1;
    int received;
    if (y_lo != 0)
    {
        MPI_Send(m_frontout, q * lx * lz, MPI_DOUBLE, comm.front, mar, comm.comm);
    }

    if (y_hi != gly - 1)
    {
        MPI_Recv(m_back, q * lx * lz, MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUS_IGNORE);
    }

    mar = 2;
    if (y_hi != gly - 1)
    {

        MPI_Isend(m_backout, q * lx * lz, MPI_DOUBLE, comm.back, mar, comm.comm, &ttttt1);
        MPI_Wait(&ttttt1, MPI_STATUS_IGNORE);
    }

    if (y_lo != 0)
    {
        MPI_Recv(m_front, q * lx * lz, MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    mar = 3;
    if (z_lo != 0)
    {
        MPI_Send(m_downout, q * lx * ly, MPI_DOUBLE, comm.down, mar, comm.comm);
    }

    if (z_hi != glz - 1)
    {
        MPI_Recv(m_up, q * lx * ly, MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUS_IGNORE);
    }

    // 12 lines
    mar = 4;
    if (z_hi != glz - 1)
    {
        MPI_Send(m_upout, q * lx * ly, MPI_DOUBLE, comm.up, mar, comm.comm);
    }

    if (z_lo != 0)
    {
        MPI_Recv(m_down, q * lx * ly, MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUS_IGNORE);
    }

    mar = 5;
    if (x_lo != 0)
    {
        MPI_Send(m_leftout, q * ly * lz, MPI_DOUBLE, comm.left, mar, comm.comm);
    }

    if (x_hi != glx - 1)
    {
        MPI_Recv(m_right, q * ly * lz, MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUS_IGNORE);
    }

    mar = 6;
    if (x_hi != glx - 1)
    {
        MPI_Send(m_rightout, q * ly * lz, MPI_DOUBLE, comm.right, mar, comm.comm);
    }

    if (x_lo != 0)
    {
        MPI_Recv(m_left, q * ly * lz, MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUS_IGNORE);
    }
    // 12 lines
    mar = 11;
    if (x_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_leftupout, q * l_l[1], MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi != glx - 1 && z_lo != 0)
        MPI_Recv(m_rightdown, q * l_l[1], MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 14;
    if (x_hi != glx - 1 && z_hi != glz - 1)
        MPI_Send(m_rightupout, q * l_l[1], MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo != 0 && z_lo != 0)
        MPI_Recv(m_leftdown, q * l_l[1], MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 13;
    if (x_lo != 0 && z_lo != 0)
        MPI_Send(m_leftdownout, q * l_l[1], MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx - 1 && z_hi != glz - 1)
        MPI_Recv(m_rightup, q * l_l[1], MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 12;

    if (x_hi != glx - 1 && z_lo != 0)
        MPI_Send(m_rightdownout, q * l_l[1], MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_leftup, q * l_l[1], MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 15;
    if (y_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_frontupout, q * l_l[0], MPI_DOUBLE, comm.frontup, mar, comm.comm);

    if (y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(m_backdown, q * l_l[0], MPI_DOUBLE, comm.backdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 17;
    if (y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontdownout, q * l_l[0], MPI_DOUBLE, comm.frontdown, mar, comm.comm);

    if (y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(m_backup, q * l_l[0], MPI_DOUBLE, comm.backup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 18;
    if (y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(m_backupout, q * l_l[0], MPI_DOUBLE, comm.backup, mar, comm.comm);

    if (y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontdown, q * l_l[0], MPI_DOUBLE, comm.frontdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 16;
    if (y_hi != gly - 1 && z_lo != 0)
        MPI_Send(m_backdownout, q * l_l[0], MPI_DOUBLE, comm.backdown, mar, comm.comm);

    if (y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_frontup, q * l_l[0], MPI_DOUBLE, comm.frontup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 7;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(m_frontleftout, q * l_l[2], MPI_DOUBLE, comm.frontleft, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Recv(m_backright, q * l_l[2], MPI_DOUBLE, comm.backright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 9;
    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Send(m_frontrightout, q * l_l[2], MPI_DOUBLE, comm.frontright, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Recv(m_backleft, q * l_l[2], MPI_DOUBLE, comm.backleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 10;
    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Send(m_backleftout, q * l_l[2], MPI_DOUBLE, comm.backleft, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Recv(m_frontright, q * l_l[2], MPI_DOUBLE, comm.frontright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Send(m_backrightout, q * l_l[2], MPI_DOUBLE, comm.backright, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(m_frontleft, q * l_l[2], MPI_DOUBLE, comm.frontleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    // 8 points
    mar = 19;
    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontleftdownout, q, MPI_DOUBLE, comm.frontleftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(m_backrightup, q, MPI_DOUBLE, comm.backrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 20;
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(m_backrightupout, q, MPI_DOUBLE, comm.backrightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontleftdown, q, MPI_DOUBLE, comm.frontleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 21;
    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(m_backleftupout, q, MPI_DOUBLE, comm.backleftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontrightdown, q, MPI_DOUBLE, comm.frontrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 22;
    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontrightdownout, q, MPI_DOUBLE, comm.frontrightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(m_backleftup, q, MPI_DOUBLE, comm.backleftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 23;
    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        MPI_Send(m_backleftdownout, q, MPI_DOUBLE, comm.backleftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_frontrightup, q, MPI_DOUBLE, comm.frontrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 24;
    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_frontrightupout, q, MPI_DOUBLE, comm.frontrightup, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(m_backleftdown, q, MPI_DOUBLE, comm.backleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 25;
    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_frontleftupout, q, MPI_DOUBLE, comm.frontleftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(m_backrightdown, q, MPI_DOUBLE, comm.backrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 26;
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        MPI_Send(m_backrightdownout, q, MPI_DOUBLE, comm.backrightdown, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_frontleftup, q, MPI_DOUBLE, comm.frontleftup, mar, comm.comm, MPI_STATUSES_IGNORE);
}
void LBM::unpack()
{
    // q faces

    if (y_lo != 0)
    {
        for (int k = 0; k < lz; k++)
        {
            for (int i = 0; i < lx; i++)
            {

                for (int ii = 0; ii < q; ii++)
                {
                    f(ii, i, l_s[1] - 1, k) = m_front[ii + i * q + k * q * lx];
                }
            }
        }
    }

    if (y_lo == 0)
    {
        for (int k = 0; k < lz; k++)
        {
            for (int i = 0; i < lx; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {

                    f(ii, i, l_e[1], k) = m_back[ii + i * q + k * q * lx];
                }
            }
        }
    }

    if (z_lo == 0)
    {
        for (int j = 0; j < ly; j++)
        {
            for (int i = 0; i < lx; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {

                    f(ii, i, j, l_e[2]) = m_up[ii + i * q + j * q * lx];
                }
            }
        }
    }

    if (z_lo != 0)
    {
        for (int j = 0; j < ly; j++)
        {
            for (int i = 0; i < lx; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {

                    f(ii, i, j, l_s[2] - 1) = m_down[ii + i * q + j * q * lx];
                }
            }
        }
    }

    if (x_hi == glx - 1)
        for (int k = 0; k < lz; k++)
        {
            for (int j = 0; j < ly; j++)
            {

                for (int ii = 0; ii < q; ii++)
                {

                    f(ii, l_s[0] - 1, j, k) = m_left[ii + j * q + k * q * ly];
                }
            }
        }

    if (x_lo == 0)
        for (int k = 0; k < lz; k++)
        {
            for (int j = 0; j < ly; j++)
            {

                for (int ii = 0; ii < q; ii++)
                {

                    f(ii, l_e[0], j, k) = m_right[ii + j * q + k * q * ly];
                }
            }
        }

    Kokkos::fence();

    if (x_hi != glx - 1 && z_lo != 0)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_e[0], j + l_s[1], l_s[2] - 1) = m_rightdown[ii + j * q];
            }
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_e[0], j + l_s[1], l_e[2]) = m_rightup[ii + j * q];
            }
        }
    }

    if (x_lo != 0 && z_hi != glz - 1)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_s[0] - 1, j + l_s[1], l_e[2]) = m_leftup[ii + j * q];
            }
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_e[0], j + l_s[1], l_e[2]) = m_rightup[ii + j * q];
            }
        }
    }

    if (y_hi != gly - 1 && z_lo != 0)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, i + l_s[0], l_e[1], l_s[2] - 1) = m_backdown[ii + i * q];
            }
        }
    }

    if (y_hi != gly - 1 && z_hi != glz - 1)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, i + l_s[0], l_e[1], l_e[2]) = m_backup[ii + i * q];
            }
        }
    }

    if (y_lo != 0 && z_lo != 0)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, i + l_s[0], l_s[1] - 1, l_s[2] - 1) = m_frontdown[ii + i * q];
            }
        }
    }

    if (y_lo != 0 && z_hi != glz - 1)
    {
        for (int i = 0; i < l_l[0]; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, i + l_s[0], l_s[1] - 1, l_e[2]) = m_frontup[ii + i * q];
            }
        }
    }

    if (x_hi != glx - 1 && y_hi != gly - 1)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_e[0], l_e[1], k + l_s[2]) = m_backright[ii + k * q];
            }
        }
    }

    if (x_lo != 0 && y_hi != gly - 1)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_s[0] - 1, l_e[1], k + l_s[2]) = m_backleft[ii + k * q];
            }
        }
    }

    if (x_hi != glx - 1 && y_lo != 0)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_e[0], l_s[1] - 1, k + l_s[2]) = m_frontright[ii + k * q];
            }
        }
    }

    if (x_lo != 0 && y_lo != 0)
    {
        for (int k = 0; k < l_l[2]; k++)
        {
            for (int ii = 0; ii < q; ii++)
            {

                f(ii, l_s[0] - 1, l_s[1] - 1, k + l_s[2]) = m_frontleft[ii + k * q];
            }
        }
    }
    Kokkos::fence();

    // 8 points
    if (x_lo != 0 && z_lo != 0 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_s[0] - 1, l_s[1] - 1, l_s[2] - 1) = m_frontleftdown[ii];
        }
    }

    if (x_hi != glx - 1 && z_lo != 0 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_e[0], l_s[1] - 1, l_s[2] - 1) = m_frontrightdown[ii];
        }
    }

    if (x_lo != 0 && z_lo != 0 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_s[0] - 1, l_e[1], l_s[2] - 1) = m_backleftdown[ii];
        }
    }

    if (x_hi != glx - 1 && z_lo != 0 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_e[0], l_e[1], l_s[2] - 1) = m_backrightdown[ii];
        }
    }

    if (x_lo != 0 && z_hi != glz - 1 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_s[0] - 1, l_s[1] - 1, l_e[2]) = m_frontleftup[ii];
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1 && y_lo != 0)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_e[0], l_s[1] - 1, l_e[2]) = m_frontrightup[ii];
        }
    }

    if (x_lo != 0 && z_hi != glz - 1 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_s[0] - 1, l_e[1], l_e[2]) = m_backleftup[ii];
        }
    }

    if (x_hi != glx - 1 && z_hi != glz - 1 && y_hi != gly - 1)
    {
        for (int ii = 0; ii < q; ii++)
        {
            f(ii, l_e[0], l_e[1], l_e[2]) = m_backrightup[ii];
        }
    }
    Kokkos::fence();
}
