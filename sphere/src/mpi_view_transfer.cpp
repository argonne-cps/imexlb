#include "lbm.hpp"
void LBM::setup_subdomain()
{

    // prepare the value needs to be transfered
    // 6 faces

    m_left = buffer_t("m_left", q, ly, lz);

    m_right = buffer_t("m_right", q, ly, lz);

    m_down = buffer_t("m_down", q, lx, ly);

    m_up = buffer_t("m_up", q, lx, ly);

    m_front = buffer_t("m_front", q, lx, lz);

    m_back = buffer_t("m_back", q, lx, lz);
    // 12 lines

    m_leftup = buffer_ut("m_leftup", q, l_l[1]);

    m_rightup = buffer_ut("m_rightup", q, l_l[1]);

    m_leftdown = buffer_ut("m_leftdown", q, l_l[1]);

    m_rightdown = buffer_ut("m_rightdown", q, l_l[1]);

    m_backleft = buffer_ut("m_backleft", q, l_l[2]);

    m_backright = buffer_ut("m_backright", q, l_l[2]);

    m_frontleft = buffer_ut("m_frontleft", q, l_l[2]);

    m_frontright = buffer_ut("m_frontdown", q, l_l[2]);

    m_backdown = buffer_ut("m_backdown", q, l_l[0]);

    m_backup = buffer_ut("m_backup", q, l_l[0]);

    m_frontdown = buffer_ut("m_frontdown", q, l_l[0]);

    m_frontup = buffer_ut("m_frontup", q, l_l[0]);

    m_frontleftdown = buffer_st("m_fld", q);

    m_frontrightdown = buffer_st("m_frd", q);

    m_frontleftup = buffer_st("m_flu", q);

    m_frontrightup = buffer_st("m_fru", q);

    m_backleftdown = buffer_st("m_bld", q);

    m_backrightdown = buffer_st("m_brd", q);

    m_backleftup = buffer_st("m_blu", q);

    m_backrightup = buffer_st("m_bru", q);

    // outdirection
    // 6 faces

    m_leftout = buffer_t("m_leftout", q, ly, lz);

    m_rightout = buffer_t("m_rightout", q, ly, lz);

    m_downout = buffer_t("m_downout", q, lx, ly);

    m_upout = buffer_t("m_upout", q, lx, ly);

    m_frontout = buffer_t("m_downout", q, lx, lz);

    m_backout = buffer_t("m_backout", q, lx, lz);

    m_leftupout = buffer_ut("m_leftupout", q, l_l[1]);

    m_rightupout = buffer_ut("m_rightupout", q, l_l[1]);

    m_leftdownout = buffer_ut("m_leftdownout", q, l_l[1]);

    m_rightdownout = buffer_ut("m_rightdownout", q, l_l[1]);

    m_backleftout = buffer_ut("m_backleftout", q, l_l[2]);

    m_backrightout = buffer_ut("m_backrightout", q, l_l[2]);

    m_frontleftout = buffer_ut("m_frontleftout", q, l_l[2]);

    m_frontrightout = buffer_ut("m_frontdownout", q, l_l[2]);

    m_backdownout = buffer_ut("m_backdownout", q, l_l[0]);

    m_backupout = buffer_ut("m_backupout", q, l_l[0]);

    m_frontdownout = buffer_ut("m_frontdownout", q, l_l[0]);

    m_frontupout = buffer_ut("m_frontupout", q, l_l[0]);

    m_frontleftdownout = buffer_st("m_fldout", q);

    m_frontrightdownout = buffer_st("m_frdout", q);

    m_frontleftupout = buffer_st("m_fluout", q);

    m_frontrightupout = buffer_st("m_fruout", q);

    m_backleftdownout = buffer_st("m_bldout", q);

    m_backrightdownout = buffer_st("m_brdout", q);

    m_backleftupout = buffer_st("m_bluout", q);

    m_backrightupout = buffer_st("m_bruout", q);
}
void LBM::pack()
{
    // 6 faces

    Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, l_s[0], Kokkos::ALL, Kokkos::ALL));

    Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, Kokkos::ALL, Kokkos::ALL));

    Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, l_s[2]));

    Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, l_e[2] - 1));

    Kokkos::deep_copy(m_frontout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, l_s[1], Kokkos::ALL));

    Kokkos::deep_copy(m_backout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, l_e[1] - 1, Kokkos::ALL));
    // 12 lines

    Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL, l_s[0], std::make_pair(l_s[1], l_e[1]), l_e[2] - 1));

    Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, std::make_pair(l_s[1], l_e[1]), l_e[2] - 1));

    Kokkos::deep_copy(m_frontleftout, Kokkos::subview(f, Kokkos::ALL, l_s[0], l_s[1], std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_frontrightout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, l_s[1], std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, l_s[0], std::make_pair(l_s[1], l_e[1]), l_s[2]));

    Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, std::make_pair(l_s[1], l_e[1]), l_s[2]));

    Kokkos::deep_copy(m_backleftout, Kokkos::subview(f, Kokkos::ALL, l_s[0], l_e[1] - 1, std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_backrightout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, l_e[1] - 1, std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_frontupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_s[1], l_e[2] - 1));

    Kokkos::deep_copy(m_frontdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_s[1], l_s[2]));

    Kokkos::deep_copy(m_backupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_e[1] - 1, l_e[2] - 1));

    Kokkos::deep_copy(m_backdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_e[1] - 1, l_s[2]));
    // 8 points

    Kokkos::deep_copy(m_frontleftdownout, Kokkos::subview(f, Kokkos::ALL, l_s[0], l_s[1], l_s[2]));

    Kokkos::deep_copy(m_frontrightdownout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, l_s[1], l_s[2]));

    Kokkos::deep_copy(m_backleftdownout, Kokkos::subview(f, Kokkos::ALL, l_s[0], l_e[1] - 1, l_s[2]));

    Kokkos::deep_copy(m_backrightdownout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, l_e[1] - 1, l_s[2]));

    Kokkos::deep_copy(m_frontleftupout, Kokkos::subview(f, Kokkos::ALL, l_s[0], l_s[1], l_e[2] - 1));

    Kokkos::deep_copy(m_frontrightupout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, l_s[1], l_e[2] - 1));

    Kokkos::deep_copy(m_backleftupout, Kokkos::subview(f, Kokkos::ALL, l_s[0], l_e[1] - 1, l_e[2] - 1));

    Kokkos::deep_copy(m_backrightupout, Kokkos::subview(f, Kokkos::ALL, l_e[0] - 1, l_e[1] - 1, l_e[2] - 1));
}

void LBM::exchange()
{
    // 6 faces
    int mar = 1;

    if (x_lo != 0)
        MPI_Send(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);

    if (x_hi != glx - 1)
        MPI_Recv(m_right.data(), m_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (x_hi != glx - 1)
        MPI_Send(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(m_left.data(), m_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 3;

    if (z_lo != 0)
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (z_hi != glz - 1)
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (z_hi != glz - 1)
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (z_lo != 0)
        MPI_Recv(m_down.data(), m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;
    if (y_lo != 0)
        MPI_Send(m_frontout.data(), m_frontout.size(), MPI_DOUBLE, comm.front, mar, comm.comm);

    if (y_hi != gly - 1)
        MPI_Recv(m_back.data(), m_back.size(), MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (y_hi != gly - 1)
        MPI_Send(m_backout.data(), m_backout.size(), MPI_DOUBLE, comm.back, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(m_front.data(), m_front.size(), MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);
    // 12 lines
    mar = 7;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(m_frontleftout.data(), m_frontleftout.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Recv(m_backright.data(), m_backright.size(), MPI_DOUBLE, comm.backright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Send(m_backrightout.data(), m_backrightout.size(), MPI_DOUBLE, comm.backright, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(m_frontleft.data(), m_frontleft.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 9;
    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Send(m_frontrightout.data(), m_frontrightout.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Recv(m_backleft.data(), m_backleft.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 10;
    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Send(m_backleftout.data(), m_backleftout.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Recv(m_frontright.data(), m_frontright.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 11;
    if (x_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi != glx - 1 && z_lo != 0)
        MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 12;

    if (x_hi != glx - 1 && z_lo != 0)
        MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 13;
    if (x_lo != 0 && z_lo != 0)
        MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx - 1 && z_hi != glz - 1)
        MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 14;
    if (x_hi != glx - 1 && z_hi != glz - 1)
        MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo != 0 && z_lo != 0)
        MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 15;
    if (y_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_frontupout.data(), m_frontupout.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm);

    if (y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(m_backdown.data(), m_backdown.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 16;
    if (y_hi != gly - 1 && z_lo != 0)
        MPI_Send(m_backdownout.data(), m_backdownout.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm);

    if (y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_frontup.data(), m_frontup.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 17;
    if (y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontdownout.data(), m_frontdownout.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm);

    if (y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(m_backup.data(), m_backup.size(), MPI_DOUBLE, comm.backup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 18;
    if (y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(m_backupout.data(), m_backupout.size(), MPI_DOUBLE, comm.backup, mar, comm.comm);

    if (y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontdown.data(), m_frontdown.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm, MPI_STATUSES_IGNORE);
    // 8 points
    mar = 19;
    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontleftdownout.data(), m_frontleftdownout.size(), MPI_DOUBLE, comm.frontleftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(m_backrightup.data(), m_backrightup.size(), MPI_DOUBLE, comm.backrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 20;
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(m_backrightupout.data(), m_backrightupout.size(), MPI_DOUBLE, comm.backrightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontleftdown.data(), m_frontleftdown.size(), MPI_DOUBLE, comm.frontleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 21;
    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Send(m_backleftupout.data(), m_backleftupout.size(), MPI_DOUBLE, comm.backleftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontrightdown.data(), m_frontrightdown.size(), MPI_DOUBLE, comm.frontrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 22;
    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontrightdownout.data(), m_frontrightdownout.size(), MPI_DOUBLE, comm.frontrightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        MPI_Recv(m_backleftup.data(), m_backleftup.size(), MPI_DOUBLE, comm.backleftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 23;
    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        MPI_Send(m_backleftdownout.data(), m_backleftdownout.size(), MPI_DOUBLE, comm.backleftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_frontrightup.data(), m_frontrightup.size(), MPI_DOUBLE, comm.frontrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 24;
    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_frontrightupout.data(), m_frontrightupout.size(), MPI_DOUBLE, comm.frontrightup, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(m_backleftdown.data(), m_backleftdown.size(), MPI_DOUBLE, comm.backleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 25;
    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        MPI_Send(m_frontleftupout.data(), m_frontleftupout.size(), MPI_DOUBLE, comm.frontleftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        MPI_Recv(m_backrightdown.data(), m_backrightdown.size(), MPI_DOUBLE, comm.backrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 26;
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        MPI_Send(m_backrightdownout.data(), m_backrightdownout.size(), MPI_DOUBLE, comm.backrightdown, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        MPI_Recv(m_frontleftup.data(), m_frontleftup.size(), MPI_DOUBLE, comm.frontleftup, mar, comm.comm, MPI_STATUSES_IGNORE);
}

void LBM::unpack()
{
    // 6 faces
    if (x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, Kokkos::ALL, Kokkos::ALL), m_left);

    if (x_hi != glx - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], Kokkos::ALL, Kokkos::ALL), m_right);

    if (z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, l_s[2] - 1), m_down);

    if (z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, l_e[2]), m_up);

    if (y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, l_s[1] - 1, Kokkos::ALL), m_front);

    if (y_hi != gly - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, l_e[1], Kokkos::ALL), m_back);
    // 12 lines
    if (x_lo != 0 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, std::make_pair(l_s[1], l_e[1]), l_e[2]), m_leftup);

    if (x_hi != glx - 1 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], std::make_pair(l_s[1], l_e[1]), l_e[2]), m_rightup);

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, l_s[1] - 1, std::make_pair(l_s[2], l_e[2])), m_frontleft);

    if (x_hi != glx - 1 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], l_s[1] - 1, std::make_pair(l_s[2], l_e[2])), m_frontright);

    if (x_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, std::make_pair(l_s[1], l_e[1]), l_s[2] - 1), m_leftdown);

    if (x_hi != glx - 1 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], std::make_pair(l_s[1], l_e[1]), l_s[2] - 1), m_rightdown);

    if (x_lo != 0 && y_hi != gly - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, l_e[1], std::make_pair(l_s[2], l_e[2])), m_backleft);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], l_e[1], std::make_pair(l_s[2], l_e[2])), m_backright);

    if (y_lo != 0 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_s[1] - 1, l_e[2]), m_frontup);

    if (y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_s[1] - 1, l_s[2] - 1), m_frontdown);

    if (y_hi != gly - 1 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_e[1], l_e[2]), m_backup);

    if (y_hi != gly - 1 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), l_e[1], l_s[2] - 1), m_backdown);
    // 8 points
    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, l_s[1] - 1, l_s[2] - 1), m_frontleftdown);
    if (x_hi != glx - 1 && y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], l_s[1] - 1, l_s[2] - 1), m_frontrightdown);
    if (x_lo != 0 && y_hi != gly - 1 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, l_e[1], l_s[2] - 1), m_backleftdown);
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], l_e[1], l_s[2] - 1), m_backrightdown);
    if (x_lo != 0 && y_lo != 0 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, l_s[1] - 1, l_e[2]), m_frontleftup);
    if (x_hi != glx - 1 && y_lo != 0 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], l_s[1] - 1, l_e[2]), m_frontrightup);
    if (x_lo != 0 && y_hi != gly - 1 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_s[0] - 1, l_e[1], l_e[2]), m_backleftup);
    if (x_hi != glx - 1 && y_hi != gly - 1 && z_hi != glz - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, l_e[0], l_e[1], l_e[2]), m_backrightup);
    
    Kokkos::fence();
};
