#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

#define pi 3.1415926
using namespace std;
void LBM::Initialize()
{

    t[0] = 8.0 / 27.0; 
    t[1] = 2.0 / 27.0;
    t[2] = 2.0 / 27.0;
    t[3] = 2.0 / 27.0;
    t[4] = 2.0 / 27.0;
    t[5] = 2.0 / 27.0;
    t[6] = 2.0 / 27.0;
    t[7] = 1.0 / 54.0;
    t[8] = 1.0 / 54.0;
    t[9] = 1.0 / 54.0;
    t[10] = 1.0 / 54.0;
    t[11] = 1.0 / 54.0;
    t[12] = 1.0 / 54.0;
    t[13] = 1.0 / 54.0;
    t[14] = 1.0 / 54.0;
    t[15] = 1.0 / 54.0;
    t[16] = 1.0 / 54.0;
    t[17] = 1.0 / 54.0;
    t[18] = 1.0 / 54.0;
    t[19] = 1.0 / 216.0;
    t[20] = 1.0 / 216.0;
    t[21] = 1.0 / 216.0;
    t[22] = 1.0 / 216.0;
    t[23] = 1.0 / 216.0;
    t[24] = 1.0 / 216.0;
    t[25] = 1.0 / 216.0;
    t[26] = 1.0 / 216.0;
    // bounce back directions
    bb[0] = 0;
    bb[1] = 2;
    bb[2] = 1;
    bb[3] = 4;
    bb[4] = 3;
    bb[5] = 6;
    bb[6] = 5;
    bb[7] = 8;
    bb[8] = 7;
    bb[9] = 10;
    bb[10] = 9;
    bb[11] = 12;
    bb[12] = 11;
    bb[13] = 14;
    bb[14] = 13;
    bb[15] = 16;
    bb[16] = 15;
    bb[17] = 18;
    bb[18] = 17;
    bb[19] = 20;
    bb[20] = 19;
    bb[21] = 22;
    bb[22] = 21;
    bb[23] = 24;
    bb[24] = 23;
    bb[25] = 26;
    bb[26] = 25;

    // discrete velocity
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;

    e[3] = 1;
    e[4] = 0;
    e[5] = 0;

    e[6] = -1;
    e[7] = 0;
    e[8] = 0;

    e[9] = 0;
    e[10] = 1;
    e[11] = 0;

    e[12] = 0;
    e[13] = -1;
    e[14] = 0;

    e[15] = 0;
    e[16] = 0;
    e[17] = 1;

    e[18] = 0;
    e[19] = 0;
    e[20] = -1;

    e[21] = 1;
    e[22] = 1;
    e[23] = 0;

    e[24] = -1;
    e[25] = -1;
    e[26] = 0;

    e[27] = 1;
    e[28] = -1;
    e[29] = 0;

    e[30] = -1;
    e[31] = 1;
    e[32] = 0;

    e[33] = 1;
    e[34] = 0;
    e[35] = 1;

    e[36] = -1;
    e[37] = 0;
    e[38] = -1;

    e[39] = 1;
    e[40] = 0;
    e[41] = -1;

    e[42] = -1;
    e[43] = 0;
    e[44] = 1;

    e[45] = 0;
    e[46] = 1;
    e[47] = 1;

    e[48] = 0;
    e[49] = -1;
    e[50] = -1;

    e[51] = 0;
    e[52] = 1;
    e[53] = -1;

    e[54] = 0;
    e[55] = -1;
    e[56] = 1;

    e[57] = 1;
    e[58] = 1;
    e[59] = 1;

    e[60] = -1;
    e[61] = -1;
    e[62] = -1;

    e[63] = 1;
    e[64] = -1;
    e[65] = 1;

    e[66] = -1;
    e[67] = 1;
    e[68] = -1;

    e[69] = 1;
    e[70] = 1;
    e[71] = -1;

    e[72] = -1;
    e[73] = -1;
    e[74] = 1;

    e[75] = 1;
    e[76] = -1;
    e[77] = -1;

    e[78] = -1;
    e[79] = 1;
    e[80] = 1;

    // macroscopic value initialization
    for (int k = 0; k < lz; k++)
    {
        for (int j = 0; j < ly; j++)
        {
            for (int i = 0; i < lx; i++)
            {    
            ua[i + j * lx + k * lx * ly] = u0  * sin((double)((double)(i - ghost + x_lo) / (double) glx * 2.0 * pi)) * 
                                                 cos((double)((double)(j - ghost + y_lo) / (double) gly * 2.0 * pi)) * 
                                                 cos((double)((double)(k - ghost + z_lo) / (double) glz * 2.0 * pi));
            va[i + j * lx + k * lx * ly] = -u0 * cos((double)((double)(i - ghost + x_lo) / (double) glx * 2.0 * pi)) * 
                                                 sin((double)((double)(j - ghost + y_lo) / (double) gly * 2.0 * pi)) * 
                                                 cos((double)((double)(k - ghost + z_lo) / (double) glz * 2.0 * pi));
            wa[i + j * lx + k * lx * ly] = 0.0;
            p[i + j * lx + k * lx * ly]  = rho0 * cs2 + rho0 * u0 * u0 / 16.0 * (cos((double)((double)(i - ghost + x_lo) / (double)glx * 2.0 * pi) * 2.0) 
                                                            +  cos((double)((double)(j - ghost + y_lo) / (double)gly * 2.0 * pi) * 2.0)) 
                                                            * (cos((double)((double)(k - ghost + z_lo) / (double)glz * 2.0 * pi) * 2.0) + 2.0);
            rho[i + j * lx + k * lx * ly] = rho0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // distribution function initialization
    for (int ii = 0; ii < q; ii++)
    {
        for (int k = ghost; k < l_e[2]; k++)
        {
            for (int j = ghost; j < l_e[1]; j++)
            {
                for (int i = ghost; i < l_e[0]; i++)
                {
                    double eu2 = pow((e[3 * ii] * ua[i + j * lx + k * lx * ly] + e[3 * ii + 1] * va[i + j * lx + k * lx * ly] + e[3 * ii + 2] * wa[i + j * lx + k * lx * ly]), 2);
                    double edu = e[3 * ii] * ua[i + j * lx + k * lx * ly] + e[3 * ii + 1] * va[i + j * lx + k * lx * ly] + e[3 * ii + 2] * wa[i + j * lx + k * lx * ly];
                    double udu = pow(ua[i + j * lx + k * lx * ly], 2) + pow(va[i + j * lx + k * lx * ly], 2) + pow(wa[i + j * lx + k * lx * ly], 2);
                    f[ii + i * q + j * q * lx + k * q * lx * ly] = t[ii] * p[i + j * lx + k * lx * ly] * 3.0 +
                                                                   t[ii] * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);
                    ft[ii + i * q + j * q * lx + k * q * lx * ly] = 0.0;
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
};
void LBM::Collision()
{
    // collision
   for (int k = l_s[2]; k < l_e[2]; k++)
    {
        for (int j = l_s[1]; j < l_e[1]; j++)
        {
            for (int i = l_s[0]; i < l_e[0]; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    double eu2 = pow((e[3 * ii] * ua[i + j * lx + k * lx * ly] + e[3 * ii + 1] * va[i + j * lx + k * lx * ly] + e[3 * ii + 2] * wa[i + j * lx + k * lx * ly]), 2);
                    double edu = e[3 * ii] * ua[i + j * lx + k * lx * ly] + e[3 * ii + 1] * va[i + j * lx + k * lx * ly] + e[3 * ii + 2] * wa[i + j * lx + k * lx * ly];
                    double udu = pow(ua[i + j * lx + k * lx * ly], 2) + pow(va[i + j * lx + k * lx * ly], 2) + pow(wa[i + j * lx + k * lx * ly], 2);

                    double feq = t[ii] * p[i + j * lx + k * lx * ly] * 3.0 +
                                 t[ii] * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);

                    f[ii + i * q + j * q * lx + k * q * lx * ly] -= (f[ii + i * q + j * q * lx + k * q * lx * ly] - feq) / (tau0 + 0.5);
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Streaming()
{

    if (y_lo == 0)
    {
        for (int k = ghost - 1; k < l_e[2] + 1; k++)
        {
            for (int i = ghost - 1; i < l_e[0] + 1; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (e[ii * 3 + 1] > 0)
                    {
                    f[ii + q * i + q * (ghost - 1) * lx + q * k * lx * ly] = f[bb[ii] + q * (i + 2 * e[ii * 3]) + q * (k + 2 * e[ii * 3 + 2]) * lx * ly + q * (ghost + 1) * lx];
                    }
                }
            }
        }
    }

    if (y_hi == gly - 1)
    {
        for (int k = ghost - 1; k < l_e[2] + 1; k++)
        {
            for (int i = ghost - 1; i < l_e[0] + 1; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (e[ii * 3 + 1] < 0)
                    {
                        f[ii + q * i + q * (l_e[1]) * lx + q * k * lx * ly] = f[bb[ii] + q * (i + 2 * e[ii * 3]) + q * (k + 2 * e[ii * 3 + 2]) * lx * ly + q * (l_e[1] - 2) * lx];
                    }
                }
            }
        }
    }
    if (z_lo == 0)
    {
        for (int j = ghost - 1; j < l_e[1] + 1; j++)
        {
            for (int i = ghost - 1; i < l_e[0] + 1; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (e[ii * 3 + 2] > 0)
                    {
                        f[ii + q * i + q * j * lx + q * (ghost - 1) * lx * ly] = f[bb[ii] + q * (i + 2 * e[ii * 3]) + q * (j + 2 * e[ii * 3 + 1]) * lx + q * (ghost + 1) * lx * ly];
                    }
                }
            }
        }
    }

    if (z_hi == glz - 1)
    {
        for (int j = ghost - 1; j < l_e[1] + 1; j++)
        {
            for (int i = ghost - 1; i < l_e[0] + 1; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (e[ii * 3 + 2] < 0)
                    {
                        f[ii + q * i + q * j * lx + q * (l_e[2]) * lx * ly] = f[bb[ii] + q * (i + 2 * e[ii * 3]) + q * (j + 2 * e[ii * 3 + 1]) * lx + q * (l_e[2] - 2) * lx * ly];
                    }
                }
            }
        }
    }

    if (x_lo == 0)
    {
        for (int k = ghost - 1; k < l_e[2] + 1; k++)
        {
            for (int j = ghost - 1; j < l_e[1] + 1; j++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (e[ii * 3] > 0)
                        f[ii + q * (ghost - 1) + q * j * lx + q * k * lx * ly] = f[ii + q * (ghost) + q * j * lx + q * k * lx * ly];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // right boundary free flow
    if (x_hi == glx - 1)
    {
        for (int k = ghost - 1; k < l_e[2] + 1; k++)
        {
            for (int j = ghost - 1; j < l_e[1] + 1; j++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (e[ii * 3] < 0)
                    {
                        f[ii + q * (l_e[0]) + q * j * lx + q * k * lx * ly] = f[ii + q * (l_e[0] - 1) + q * (j + e[ii * 3 + 1]) * lx + q * (k + e[ii * 3 + 2]) * lx * ly];
                    }
                }
            }
        }
    }
    // streaming process
    MPI_Barrier(MPI_COMM_WORLD);
    for (int k = ghost; k < l_e[2]; k++)
    {
        for (int j = ghost; j < l_e[1]; j++)
        {
            for (int i = ghost; i < l_e[0]; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    if (usr[i + j * lx + k * lx * ly] == 0 && usr[i + e[ii * dim] + (j + e[ii * dim + 1]) * lx + (k + e[ii * dim + 2]) * lx * ly] == 1)
                    {
                        f[ii + i * q + q * j * lx + q * k * lx * ly] = f[bb[ii] + (i + 2 * e[ii * dim]) * q + q * (j + 2 * e[ii * dim + 1]) * lx + q * (k + 2 * e[ii * dim + 2]) * lx * ly];
                    }
                }
            }
        }
    }

    for (int k = ghost; k < l_e[2]; k++)
    {
        for (int j = ghost; j < l_e[1]; j++)
        {
            for (int i = ghost; i < l_e[0]; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    ft[ii + i * q + q * j * lx + q * k * lx * ly] = f[ii + (i - e[ii * dim]) * q + q * (j - e[ii * dim + 1]) * lx + q * (k - e[ii * dim + 2]) * lx * ly];
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int k = ghost; k < l_e[2]; k++)
    {
        for (int j = ghost; j < l_e[1]; j++)
        {
            for (int i = ghost; i < l_e[0]; i++)
            {
                for (int ii = 0; ii < q; ii++)
                {
                    f[ii + i * q + q * j * lx + q * k * lx * ly] = ft[ii + i * q + j * q * lx + q * k * lx * ly];
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Update()
{
    // update macroscopic value
    for (int k = ghost; k < l_e[2]; k++)
    {
        for (int j = ghost; j < l_e[1]; j++)
        {
            for (int i = ghost; i < l_e[0]; i++)
            {
                ua[i + j * lx + k * lx * ly] = 0.0;
                va[i + j * lx + k * lx * ly] = 0.0;
                wa[i + j * lx + k * lx * ly] = 0.0;
                p[i + j * lx + k * lx * ly] = 0.0;
                for (int ii = 0; ii < q; ii++)
                {

                    p[i + j * lx + k * lx * ly] = p[i + j * lx + k * lx * ly] + f[ii + i * q + j * lx * q + k * lx * ly * q] / 3.0;
                    ua[i + j * lx + k * lx * ly] = ua[i + j * lx + k * lx * ly] + f[ii + i * q + j * lx * q + k * lx * ly * q] * e[ii * 3];
                    va[i + j * lx + k * lx * ly] = va[i + j * lx + k * lx * ly] + f[ii + i * q + j * lx * q + k * lx * ly * q] * e[ii * 3 + 1];
                    wa[i + j * lx + k * lx * ly] = wa[i + j * lx + k * lx * ly] + f[ii + i * q + j * lx * q + k * lx * ly * q] * e[ii * 3 + 2];
                }
                ua[i + j * lx + k * lx * ly] = ua[i + j * lx + k * lx * ly] * usr[i + j * lx + k * lx * ly];
                va[i + j * lx + k * lx * ly] = va[i + j * lx + k * lx * ly] * usr[i + j * lx + k * lx * ly];
                wa[i + j * lx + k * lx * ly] = wa[i + j * lx + k * lx * ly] * usr[i + j * lx + k * lx * ly];
                p[i + j * lx + k * lx * ly] = p[i + j * lx + k * lx * ly] * usr[i + j * lx + k * lx * ly];

                if (x_lo == 0)
                {
                    ua[ghost + j * lx + k * lx * ly] = u0 * 4.0 * (z_lo + k - ghost) * (glz - 1 - (z_lo + k - ghost)) / pow((glz - 1), 2);
                    va[ghost + j * lx + k * lx * ly] = 0.0;
                    wa[ghost + j * lx + k * lx * ly] = 0.0;
                }
                if (x_hi == glx - 1)
                {
                    p[l_e[0] - 1 + j * lx + k * lx * ly] = 0.0;
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, wmin, wmax, vmin, vmax, pmin, pmax;
    double uumin, uumax, wwmin, wwmax, vvmin, vvmax, ppmin, ppmax;

    // transfer
    double *uu, *vv, *ww, *pp, *xx, *yy, *zz;

    uu = (double *)malloc(ex * ey * ez * sizeof(double));
    vv = (double *)malloc(ex * ey * ez * sizeof(double));
    ww = (double *)malloc(ex * ey * ez * sizeof(double));
    pp = (double *)malloc(ex * ey * ez * sizeof(double));
    xx = (double *)malloc(ex * ey * ez * sizeof(double));
    yy = (double *)malloc(ex * ey * ez * sizeof(double));
    zz = (double *)malloc(ex * ey * ez * sizeof(double));

    for (int k = 0; k < ez; k++)
    {
        for (int j = 0; j < ey; j++)
        {
            for (int i = 0; i < ex; i++)
            {

                uu[i + j * ex + k * ey * ex] = ua[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                vv[i + j * ex + k * ey * ex] = va[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                ww[i + j * ex + k * ey * ex] = wa[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                pp[i + j * ex + k * ey * ex] = p[i + ghost + (j + ghost) * lx + (k + ghost) * lx * ly];
                xx[i + j * ex + k * ey * ex] = (double)4.0 * (x_lo + i) / (glx - 1);
                yy[i + j * ex + k * ey * ex] = (double)(y_lo + j) / (gly - 1);
                zz[i + j * ex + k * ey * ex] = (double)(z_lo + k) / (glz - 1);
            }
        }
    }

    umin = *min_element(uu, uu + ex * ey * ez - 1);
    umax = *max_element(uu, uu + ex * ey * ez - 1);
    vmin = *min_element(vv, vv + ex * ey * ez - 1);
    vmax = *max_element(vv, vv + ex * ey * ez - 1);
    wmin = *min_element(ww, ww + ex * ey * ez - 1);
    wmax = *max_element(ww, ww + ex * ey * ez - 1);

    pmin = *min_element(pp, pp + ex * ey * ez - 1);
    pmax = *max_element(pp, pp + ex * ey * ez - 1);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&wmin, &wwmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wmax, &wwmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&pmin, &ppmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pmax, &ppmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    if (comm.me == 0)
    {

        MPI_File_seek(fh, offset, MPI_SEEK_SET);
        // header !version number
        MPI_File_write(fh, version, 8, MPI_CHAR, &status);
        // INTEGER 1
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 3*4+8=20
        // variable name
        tp = 7;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 122;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 117;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 118;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 119;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+15*4=80
        // Zone Marker
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // Zone Name
        tp = 90;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 79;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 78;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 69;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 32;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 49;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 80 + 10 * 4 = 120

        // Strand id
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE COLOR
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE TYPE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SPECIFY VAR LOCATION
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ARE RAW LOCAL
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // NUMBER OF MISCELLANEOUS
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ORDERED ZONE
        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = glz;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 120 + 13 * 4 = 172
        // EOHMARKER
        ttp = 357.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // DATA SECTION
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // VARIABLE DATA FORMAT
        tp = 2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // PASSIVE VARIABLE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SHARING VARIABLE
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE NUMBER
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 172 + 12 * 4 = 220
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 4.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wwmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wwmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 220 + 14 * 8 = 332
    }

    offset = 332;

    int glolen[3] = {glx, gly, glz};
    int localstart[3] = {x_lo, y_lo, z_lo};

    MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    // MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(7, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, zz, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, ww, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, pp, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(ww);
    free(pp);
    free(xx);
    free(yy);
    free(zz);

    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,z,f" << std::endl;
    outfile << "zone I=" << lx - 6 << ",J=" << ly - 6 << ",K=" << lz - 6 << std::endl;

    for (int k = 3; k < lz - 3; k++)
    {
        for (int j = 3; j < ly - 3; j++)
        {
            for (int i = 3; i < lx - 3; i++)
            {

                //outfile << std::setprecision(8) << setiosflags(std::ios::left) << x_lo + i - 3 << " " << y_lo + j - 3 << " " << z_lo + k - 3 << " " << f(0, i, j, k) << std::endl;
            }
        }
    }

    outfile.close();
    if (comm.me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};
