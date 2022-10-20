#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#define pi 3.1415926
using namespace std;
void LBM::Initialize()
{

    f = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("f", q, lx, ly, lz);
    ft = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("ft", q, lx, ly, lz);
    fb = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("fb", q, lx, ly, lz);

    ua = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("u", lx, ly, lz);
    va = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("v", lx, ly, lz);
    wa = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("v", lx, ly, lz);
    rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("rho", lx, ly, lz);
    p = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("p", lx, ly, lz);

    e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);
    usr = Kokkos::View<int ***, Kokkos::CudaUVMSpace>("usr", lx, ly, lz);
    ran = Kokkos::View<int ***, Kokkos::CudaUVMSpace>("ran", lx, ly, lz);
    bb = Kokkos::View<int *, Kokkos::CudaUVMSpace>("b", q);

    //  weight function
    t(0) = 8.0 / 27.0;
    t(1) = 2.0 / 27.0;
    t(2) = 2.0 / 27.0;
    t(3) = 2.0 / 27.0;
    t(4) = 2.0 / 27.0;
    t(5) = 2.0 / 27.0;
    t(6) = 2.0 / 27.0;
    t(7) = 1.0 / 54.0;
    t(8) = 1.0 / 54.0;
    t(9) = 1.0 / 54.0;
    t(10) = 1.0 / 54.0;
    t(11) = 1.0 / 54.0;
    t(12) = 1.0 / 54.0;
    t(13) = 1.0 / 54.0;
    t(14) = 1.0 / 54.0;
    t(15) = 1.0 / 54.0;
    t(16) = 1.0 / 54.0;
    t(17) = 1.0 / 54.0;
    t(18) = 1.0 / 54.0;
    t(19) = 1.0 / 216.0;
    t(20) = 1.0 / 216.0;
    t(21) = 1.0 / 216.0;
    t(22) = 1.0 / 216.0;
    t(23) = 1.0 / 216.0;
    t(24) = 1.0 / 216.0;
    t(25) = 1.0 / 216.0;
    t(26) = 1.0 / 216.0;
    // bounce back directions
    bb(0) = 0;
    bb(1) = 2;
    bb(2) = 1;
    bb(3) = 4;
    bb(4) = 3;
    bb(5) = 6;
    bb(6) = 5;
    bb(7) = 8;
    bb(8) = 7;
    bb(9) = 10;
    bb(10) = 9;
    bb(11) = 12;
    bb(12) = 11;
    bb(13) = 14;
    bb(14) = 13;
    bb(15) = 16;
    bb(16) = 15;
    bb(17) = 18;
    bb(18) = 17;
    bb(19) = 20;
    bb(20) = 19;
    bb(21) = 22;
    bb(22) = 21;
    bb(23) = 24;
    bb(24) = 23;
    bb(25) = 26;
    bb(26) = 25;

    // discrete velocity
    e(0, 0) = 0;
    e(0, 1) = 0;
    e(0, 2) = 0;

    e(1, 0) = 1;
    e(1, 1) = 0;
    e(1, 2) = 0;

    e(2, 0) = -1;
    e(2, 1) = 0;
    e(2, 2) = 0;

    e(3, 0) = 0;
    e(3, 1) = 1;
    e(3, 2) = 0;

    e(4, 0) = 0;
    e(4, 1) = -1;
    e(4, 2) = 0;

    e(5, 0) = 0;
    e(5, 1) = 0;
    e(5, 2) = 1;

    e(6, 0) = 0;
    e(6, 1) = 0;
    e(6, 2) = -1;

    e(7, 0) = 1;
    e(7, 1) = 1;
    e(7, 2) = 0;

    e(8, 0) = -1;
    e(8, 1) = -1;
    e(8, 2) = 0;

    e(9, 0) = 1;
    e(9, 1) = -1;
    e(9, 2) = 0;

    e(10, 0) = -1;
    e(10, 1) = 1;
    e(10, 2) = 0;

    e(11, 0) = 1;
    e(11, 1) = 0;
    e(11, 2) = 1;

    e(12, 0) = -1;
    e(12, 1) = 0;
    e(12, 2) = -1;

    e(13, 0) = 1;
    e(13, 1) = 0;
    e(13, 2) = -1;

    e(14, 0) = -1;
    e(14, 1) = 0;
    e(14, 2) = 1;

    e(15, 0) = 0;
    e(15, 1) = 1;
    e(15, 2) = 1;

    e(16, 0) = 0;
    e(16, 1) = -1;
    e(16, 2) = -1;

    e(17, 0) = 0;
    e(17, 1) = 1;
    e(17, 2) = -1;

    e(18, 0) = 0;
    e(18, 1) = -1;
    e(18, 2) = 1;

    e(19, 0) = 1;
    e(19, 1) = 1;
    e(19, 2) = 1;

    e(20, 0) = -1;
    e(20, 1) = -1;
    e(20, 2) = -1;

    e(21, 0) = 1;
    e(21, 1) = -1;
    e(21, 2) = 1;

    e(22, 0) = -1;
    e(22, 1) = 1;
    e(22, 2) = -1;

    e(23, 0) = 1;
    e(23, 1) = 1;
    e(23, 2) = -1;

    e(24, 0) = -1;
    e(24, 1) = -1;
    e(24, 2) = 1;

    e(25, 0) = 1;
    e(25, 1) = -1;
    e(25, 2) = -1;

    e(26, 0) = -1;
    e(26, 1) = 1;
    e(26, 2) = 1;

    // macroscopic value initialization

    // macroscopic value initialization
    Kokkos::parallel_for(
        "initialize", mdrange_policy3({0, 0, 0}, {lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            ua(i, j, k) = u0  * sin((double)((double)(i - ghost + x_lo) / (double) glx * 2.0 * pi)) * cos((double)((double)(j - ghost + y_lo) / (double) gly * 2.0 * pi))
                              * cos((double)((double)(k - ghost + z_lo) / (double) glz * 2.0 * pi));
            va(i, j, k) = -u0 * cos((double)((double)(i - ghost + x_lo) / (double) glx * 2.0 * pi)) * sin((double)((double)(j - ghost + y_lo) / (double) gly * 2.0 * pi)) 
                              * cos((double)((double)(k - ghost + z_lo) / (double) glz * 2.0 * pi));
            wa(i, j, k) = 0;
            p(i, j, k) = rho0 * cs2 + rho0 * u0 * u0 / 16.0 * (cos((double)((double)(i - ghost + x_lo) / (double)glx * 2.0 * pi) * 2.0) 
                                                            +  cos((double)((double)(j - ghost + y_lo) / (double)gly * 2.0 * pi) * 2.0)) 
                                                            * (cos((double)((double)(k - ghost + z_lo) / (double)glz * 2.0 * pi) * 2.0) + 2.0);
            rho(i, j, k) = rho0;
        });

    // distribution function initialization
    Kokkos::parallel_for(
        "initf", mdrange_policy4({0, 0, 0, 0}, {q, lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            double edu = e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k);
            double udu = pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2);
            double eu2 = pow((e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)), 2);

            f(ii, i, j, k) = t(ii) * p(i, j, k) * 3.0 + t(ii) * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);

            ft(ii, i, j, k) = 0;
        });

    Kokkos::fence();
};
void LBM::Collision()
{
    // collision

    Kokkos::parallel_for(
        "collision", mdrange_policy4({0, l_s[0], l_s[1], l_s[2]}, {q, l_e[0], l_e[1], l_e[2]}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            double edu = e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k);
            double udu = pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2);
            double eu2 = pow((e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)), 2);

            double feq = t(ii) * p(i, j, k) * 3.0 + t(ii) * (3.0 * edu + 4.5 * eu2 - 1.5 * udu);

            f(ii, i, j, k) -= (f(ii, i, j, k) - feq) / (tau0 + 0.5);
        });
    Kokkos::fence();
};

void LBM::Streaming()
{

    // streaming process
    Kokkos::parallel_for(
        "stream1", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            ft(ii, i, j, k) = f(ii, i - e(ii, 0), j - e(ii, 1), k - e(ii, 2));
        });

    Kokkos::fence();

    Kokkos::parallel_for(
        "stream2", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            f(ii, i, j, k) = ft(ii, i, j, k);
        });

    Kokkos::fence();
};

void LBM::Update1()
{
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::parallel_for(
        "update", team_policy(lz, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int k = team_member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, lx * ly), [&](const int &ij)
                {
                    const int i = ij % lx;
                    const int j = ij / lx;
                    p(i, j, k) = 0.0;


                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &pm)
                        { pm += f(ii, i, j, k) / 3.0; },
                        p(i, j, k));



                     });
        });


    Kokkos::fence();
};

void LBM::Update()
{
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::parallel_for(
        "update", team_policy(lz, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int k = team_member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, lx * ly), [&](const int &ij)
                {
                    const int i = ij % lx;
                    const int j = ij / lx;
                    ua(i, j, k) = 0.0;
                    va(i, j, k) = 0.0;
                    wa(i, j, k) = 0.0;
                    p(i, j, k) = 0.0;


                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &um)
                        { um += f(ii, i, j, k) * e(ii, 0); },
                        ua(i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &vm)
                        { vm += f(ii, i, j, k) * e(ii, 1); },
                        va(i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &wm)
                        { wm += f(ii, i, j, k) * e(ii, 2); },
                        wa(i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &pm)
                        { pm += f(ii, i, j, k) / 3.0; },
                        p(i, j, k));



                     });
        });
    Kokkos::fence();

};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPIO_Request request;
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
    int start[3];
    uu = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    vv = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    ww = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    pp = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    xx = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    yy = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    zz = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));

    for (int k = 0; k < l_l[2]; k++)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int i = 0; i < l_l[0]; i++)
            {

                uu[i + j * l_l[0] + k * l_l[1] * l_l[0]] = ua(i + ghost, j + ghost, k + ghost);
                vv[i + j * l_l[0] + k * l_l[1] * l_l[0]] = va(i + ghost, j + ghost, k + ghost);
                ww[i + j * l_l[0] + k * l_l[1] * l_l[0]] = wa(i + ghost, j + ghost, k + ghost);
                pp[i + j * l_l[0] + k * l_l[1] * l_l[0]] = p(i + ghost, j + ghost, k + ghost);
                xx[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(x_lo + i) / (glx - 1);
                yy[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(y_lo + j) / (gly - 1);
                zz[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(z_lo + k) / (glz - 1);
            }
        }
    }

        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(wmax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(pmax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(wmin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(pmin));
        Kokkos::fence();
        std::string str1 = "output" + std::to_string(n) + ".plt";
        const char *na = str1.c_str();
        std::string str2 = "#!TDV112";
        const char *version = str2.c_str();
        MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

        MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&wmin, &wwmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&wmax, &wwmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&pmin, &ppmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&pmax, &ppmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
            fp = 1.0;
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
        int iniarr[3] = {0, 0, 0};
        int localstart[3] = {x_lo, y_lo, z_lo};
        MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

        MPI_Type_commit(&DATATYPE);

        MPI_Type_contiguous(7, DATATYPE, &FILETYPE);

        MPI_Type_commit(&FILETYPE);

        MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

        MPI_File_write_all(fh, xx, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, yy, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, zz, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, uu, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, vv, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, ww, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, pp, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_close(&fh);


        if (comm.me == 0)
        {

        
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
        }

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

                outfile << std::setprecision(8) << setiosflags(std::ios::left) << x_lo + i - 3 << " " << y_lo + j - 3 << " " << z_lo + k - 3 << " " << f(0, i, j, k) << std::endl;
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

/*Kokkos::View<double****,Kokkos::CudaUVMSpace> LBM::d_c(Kokkos::View<double***,Kokkos::CudaUVMSpace> c)
{
    Kokkos::View<double ****, Kokkos::CudaUVMSpace> dc= Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dc_", dim, lx, ly, lz);
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;


    Kokkos::parallel_for(
        "dc", team_policy(ly-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            dc(0, i, j) = 0.0;
                            dc(1, i, j) = 0.0;

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int& ii, double &dc0_tem) {
                        dc0_tem += t(ii) *  e(ii, 0) * (c(i + e(ii, 0), j + e(ii, 1)) - c(i - e(ii, 0), j - e(ii, 1))) / 2.0 / cs2;},
                        dc(0,i,j));

                        Kokkos::parallel_reduce(
                                         Kokkos::ThreadVectorRange(team_member, q),[&](const int& ii, double &dc1_tem) {
                        dc1_tem += t(ii) *  e(ii, 1) * (c(i + e(ii, 0), j + e(ii, 1)) - c(i - e(ii, 0), j - e(ii, 1))) / 2.0 / cs2;},
                        dc(1,i,j));


             }); });

Kokkos::fence();
    return dc;
};*/
