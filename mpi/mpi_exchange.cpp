#include "lbm.hpp"

void LBM::exchange()
{
    int mar = 1;
    if (y_lo != 0)
        MPI_Send(&f[l_s[1] * lx * q], 1, m_face[1], comm.front, mar, comm.comm);

    if (y_hi != gly - 1)
        MPI_Recv(&f[l_e[1] * lx * q], 1, m_face[1], comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (y_hi != gly - 1)
        MPI_Send(&f[(l_e[1] - 1) * lx * q], 1, m_face[1], comm.back, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(&f[(l_s[1] - 1) * lx * q], 1, m_face[1], comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 3;
    if (z_lo != 0)
        MPI_Send(&f[l_s[2] * lx * ly * q], 1, m_face[0], comm.down, mar, comm.comm);

    if (z_hi != glz - 1)
        MPI_Recv(&f[l_e[2] * lx * ly * q], 1, m_face[0], comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (z_hi != glz - 1)
        MPI_Send(&f[(l_e[2] - 1) * lx * ly * q], 1, m_face[0], comm.up, mar, comm.comm);

    if (z_lo != 0)
        MPI_Recv(&f[(l_s[2] - 1) * lx * ly * q], 1, m_face[0], comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;
    if (x_lo != 0)
        MPI_Send(&f[l_s[0] * q], 1, m_face[2], comm.left, mar, comm.comm);

    if (x_hi != glx - 1)
        MPI_Recv(&f[l_e[0] * q], 1, m_face[2], comm.right, mar, comm.comm, MPI_STATUS_IGNORE);

    mar = 6;
    if (x_hi != glx - 1)
        MPI_Send(&f[(l_e[0] - 1) * q], 1, m_face[2], comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(&f[(l_s[0] - 1) * q], 1, m_face[2], comm.left, mar, comm.comm, MPI_STATUS_IGNORE);

    // 12 lines
    mar = 7;
    if (x_lo != 0 && z_hi != glz - 1)
        MPI_Send(&f[l_s[0] * q + (l_e[2] - 1) * lx * ly * q], 1, m_line[1], comm.leftup, mar, comm.comm);

    if (x_hi != glx - 1 && z_lo != 0)
        MPI_Recv(&f[(l_e[0]) * q + (l_s[2] - 1) * lx * ly * q], 1, m_line[1], comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx - 1 && z_hi != glz - 1)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_e[2] - 1) * lx * ly * q], 1, m_line[1], comm.rightup, mar, comm.comm);

    if (x_lo != 0 && z_lo != 0)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_s[2] - 1) * lx * ly * q], 1, m_line[1], comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 9;
    if (x_lo != 0 && z_lo != 0)
        MPI_Send(&f[(l_s[0]) * q + (l_s[2]) * lx * ly * q], 1, m_line[1], comm.leftdown, mar, comm.comm);

    if (x_hi != glx - 1 && z_hi != glz - 1)
        MPI_Recv(&f[(l_e[0]) * q + (l_e[2]) * lx * ly * q], 1, m_line[1], comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 10;

    if (x_hi != glx - 1 && z_lo != 0)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_s[2]) * lx * ly * q], 1, m_line[1], comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && z_hi != glz - 1)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_e[2]) * lx * ly * q], 1, m_line[1], comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 11;
    if (y_lo != 0 && z_hi != glz - 1)
        MPI_Send(&f[(l_s[1]) * q * lx + (l_e[2] - 1) * lx * ly * q], 1, m_line[0], comm.frontup, mar, comm.comm);

    if (y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(&f[(l_e[1]) * q * lx + (l_s[2] - 1) * lx * ly * q], 1, m_line[0], comm.backdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 12;
    if (y_lo != 0 && z_lo != 0)
        MPI_Send(&f[(l_s[1]) * q * lx + (l_s[2]) * lx * ly * q], 1, m_line[0], comm.frontdown, mar, comm.comm);

    if (y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(&f[(l_e[1]) * q * lx + (l_e[2]) * lx * ly * q], 1, m_line[0], comm.backup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 13;
    if (y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(&f[(l_e[1] - 1) * q * lx + (l_e[2] - 1) * lx * ly * q], 1, m_line[0], comm.backup, mar, comm.comm);

    if (y_lo != 0 && z_lo != 0)
        MPI_Recv(&f[(l_s[1] - 1) * q * lx + (l_s[2] - 1) * lx * ly * q], 1, m_line[0], comm.frontdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 14;
    if (y_hi != gly - 1 && z_lo != 0)
        MPI_Send(&f[(l_e[1] - 1) * q * lx + (l_s[2]) * lx * ly * q], 1, m_line[0], comm.backdown, mar, comm.comm);

    if (y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(&f[(l_s[1] - 1) * q * lx + (l_e[2]) * lx * ly * q], 1, m_line[0], comm.frontup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 15;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(&f[(l_s[0]) * q + (l_s[1]) * q * lx], 1, m_line[2], comm.frontleft, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Recv(&f[(l_e[0]) * q + (l_e[1]) * q * lx], 1, m_line[2], comm.backright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 16;
    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_s[1]) * q * lx], 1, m_line[2], comm.frontright, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_e[1]) * q * lx], 1, m_line[2], comm.backleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 17;
    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Send(&f[(l_s[0]) * q + (l_e[1] - 1) * q * lx], 1, m_line[2], comm.backleft, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Recv(&f[(l_e[0]) * q + (l_s[1] - 1) * q * lx], 1, m_line[2], comm.frontright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 18;
    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_e[1] - 1) * q * lx], 1, m_line[2], comm.backright, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_s[1] - 1) * q * lx], 1, m_line[2], comm.frontleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    // 8 points
    mar = 19;
    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Send(&f[(l_s[0]) * q + (l_s[1]) * q * lx + (l_s[2]) * q * lx * ly], 1, m_point, comm.frontleftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(&f[(l_e[0]) * q + (l_e[1]) * q * lx + (l_e[2]) * q * lx * ly], 1, m_point, comm.backrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 20;
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_e[1] - 1) * q * lx + (l_e[2] - 1) * q * lx * ly], 1, m_point, comm.backrightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_s[1] - 1) * q * lx + (l_s[2] - 1) * q * lx * ly], 1, m_point, comm.frontleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 21;
    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(&f[(l_s[0]) * q + (l_e[1] - 1) * q * lx + (l_e[2] - 1) * q * lx * ly], 1, m_point, comm.backleftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        MPI_Recv(&f[(l_e[0]) * q + (l_s[1] - 1) * q * lx + (l_s[2] - 1) * q * lx * ly], 1, m_point, comm.frontrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 22;
    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_s[1]) * q * lx + (l_s[2]) * q * lx * ly], 1, m_point, comm.frontrightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_e[1]) * q * lx + (l_e[2]) * q * lx * ly], 1, m_point, comm.backleftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 23;
    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        MPI_Send(&f[(l_s[0]) * q + (l_e[1] - 1) * q * lx + (l_s[2]) * q * lx * ly], 1, m_point, comm.backleftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(&f[(l_e[0]) * q + (l_s[1] - 1) * q * lx + (l_e[2]) * q * lx * ly], 1, m_point, comm.frontrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 24;
    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_s[1]) * q * lx + (l_e[2] - 1) * q * lx * ly], 1, m_point, comm.frontrightup, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_e[1]) * q * lx + (l_s[2] - 1) * q * lx * ly], 1, m_point, comm.backleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 25;
    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        MPI_Send(&f[(l_s[0]) * q + (l_s[1]) * q * lx + (l_e[2] - 1) * q * lx * ly], 1, m_point, comm.frontleftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(&f[(l_e[0]) * q + (l_e[1]) * q * lx + (l_s[2] - 1) * q * lx * ly], 1, m_point, comm.backrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 26;
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        MPI_Send(&f[(l_e[0] - 1) * q + (l_e[1] - 1) * q * lx + (l_s[2]) * q * lx * ly], 1, m_point, comm.backrightdown, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(&f[(l_s[0] - 1) * q + (l_s[1] - 1) * q * lx + (l_e[2]) * q * lx * ly], 1, m_point, comm.frontleftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
}
