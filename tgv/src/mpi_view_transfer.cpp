#include "lbm.hpp"
void LBM::setup_subdomain()
{

    // prepare the value needs to be transfered
    // 6 faces

    m_left = buffer_ft("m_left", q, ghost, ly, lz);

    m_right = buffer_ft("m_right", q, ghost, ly, lz);

    m_down = buffer_ft("m_down", q, lx, ly, ghost);

    m_up = buffer_ft("m_up", q, lx, ly, ghost);

    m_front = buffer_ft("m_front", q, lx, ghost, lz);

    m_back = buffer_ft("m_back", q, lx, ghost, lz);
    // 12 lines

    m_leftup = buffer_ft("m_leftup", q, ghost, l_l[1], ghost);

    m_rightup = buffer_ft("m_rightup", q, ghost, l_l[1], ghost);

    m_leftdown = buffer_ft("m_leftdown", q, ghost, l_l[1], ghost);

    m_rightdown = buffer_ft("m_rightdown", q, ghost, l_l[1], ghost);

    m_backleft = buffer_ft("m_backleft", q, ghost, ghost, l_l[2]);

    m_backright = buffer_ft("m_backright", q, ghost, ghost, l_l[2]);

    m_frontleft = buffer_ft("m_frontleft", q, ghost, ghost, l_l[2]);

    m_frontright = buffer_ft("m_frontdown", q, ghost, ghost, l_l[2]);

    m_backdown = buffer_ft("m_backdown", q, l_l[0], ghost, ghost);

    m_backup = buffer_ft("m_backup", q, l_l[0], ghost, ghost);

    m_frontdown = buffer_ft("m_frontdown", q, l_l[0], ghost, ghost);

    m_frontup = buffer_ft("m_frontup", q, l_l[0], ghost, ghost);

    m_frontleftdown = buffer_ft("m_fld", q, ghost, ghost, ghost);

    m_frontrightdown = buffer_ft("m_frd", q, ghost, ghost, ghost);

    m_frontleftup = buffer_ft("m_flu", q, ghost, ghost, ghost);

    m_frontrightup = buffer_ft("m_fru", q, ghost, ghost, ghost);

    m_backleftdown = buffer_ft("m_bld", q, ghost, ghost, ghost);

    m_backrightdown = buffer_ft("m_brd", q, ghost, ghost, ghost);

    m_backleftup = buffer_ft("m_blu", q, ghost, ghost, ghost);

    m_backrightup = buffer_ft("m_bru", q, ghost, ghost, ghost);

    // outdirection
    // 6 faces

    m_leftout = buffer_ft("m_leftout", q, ghost, ly, lz);

    m_rightout = buffer_ft("m_rightout", q, ghost, ly, lz);

    m_downout = buffer_ft("m_downout", q, lx, ly, ghost);

    m_upout = buffer_ft("m_upout", q, lx, ly, ghost);

    m_frontout = buffer_ft("m_downout", q, lx, ghost, lz);

    m_backout = buffer_ft("m_backout", q, lx, ghost, lz);

    m_leftupout = buffer_ft("m_leftupout", q, ghost, l_l[1], ghost);

    m_rightupout = buffer_ft("m_rightupout", q, ghost, l_l[1], ghost);

    m_leftdownout = buffer_ft("m_leftdownout", q, ghost, l_l[1], ghost);

    m_rightdownout = buffer_ft("m_rightdownout", q, ghost, l_l[1], ghost);

    m_backleftout = buffer_ft("m_backleftout", q, ghost, ghost, l_l[2]);

    m_backrightout = buffer_ft("m_backrightout", q, ghost, ghost, l_l[2]);

    m_frontleftout = buffer_ft("m_frontleftout", q, ghost, ghost, l_l[2]);

    m_frontrightout = buffer_ft("m_frontdownout", q, ghost, ghost, l_l[2]);

    m_backdownout = buffer_ft("m_backdownout", q, l_l[0], ghost, ghost);

    m_backupout = buffer_ft("m_backupout", q, l_l[0], ghost, ghost);

    m_frontdownout = buffer_ft("m_frontdownout", q, l_l[0], ghost, ghost);

    m_frontupout = buffer_ft("m_frontupout", q, l_l[0], ghost, ghost);

    m_frontleftdownout = buffer_ft("m_fldout", q, ghost, ghost, ghost);

    m_frontrightdownout = buffer_ft("m_frdout", q, ghost, ghost, ghost);

    m_frontleftupout = buffer_ft("m_fluout", q, ghost, ghost, ghost);

    m_frontrightupout = buffer_ft("m_fruout", q, ghost, ghost, ghost);

    m_backleftdownout = buffer_ft("m_bldout", q, ghost, ghost, ghost);

    m_backrightdownout = buffer_ft("m_brdout", q, ghost, ghost, ghost);

    m_backleftupout = buffer_ft("m_bluout", q, ghost, ghost, ghost);

    m_backrightupout = buffer_ft("m_bruout", q, ghost, ghost, ghost);
}
void LBM::pack()
{
    // 6 faces

    Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), Kokkos::ALL, Kokkos::ALL));

    Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), Kokkos::ALL, Kokkos::ALL));

    Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_frontout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), Kokkos::ALL));

    Kokkos::deep_copy(m_backout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, std::make_pair(ly - 2 * ghost, ly - ghost), Kokkos::ALL));
    // 12 lines

    Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_frontleftout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost), std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_frontrightout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ghost, 2 * ghost), std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_backleftout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_backrightout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(l_s[2], l_e[2])));

    Kokkos::deep_copy(m_frontupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ghost, 2 * ghost), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_frontdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_backupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_backdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(ghost, 2 * ghost)));
    // 8 points

    Kokkos::deep_copy(m_frontleftdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_frontrightdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_backleftdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_backrightdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(ghost, 2 * ghost)));

    Kokkos::deep_copy(m_frontleftupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_frontrightupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ghost, 2 * ghost), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_backleftupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(lz - 2 * ghost, lz - ghost)));

    Kokkos::deep_copy(m_backrightupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ly - 2 * ghost, ly - ghost), std::make_pair(lz - 2 * ghost, lz - ghost)));
}

void LBM::exchange()
{

int mar = 1;
        MPI_Isend(m_leftout.data()           , m_leftout.size()          , MPI_DOUBLE, comm.left            , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_right.data()             , m_right.size()            , MPI_DOUBLE, comm.right           , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 2;
        MPI_Isend(m_rightout.data()          , m_rightout.size()         , MPI_DOUBLE, comm.right           , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_left.data()              , m_left.size()             , MPI_DOUBLE, comm.left            , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 3;
        MPI_Isend(m_downout.data()           , m_downout.size()          , MPI_DOUBLE, comm.down            , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_up.data()                , m_up.size()               , MPI_DOUBLE, comm.up              , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 4;
        MPI_Isend(m_upout.data()             , m_upout.size()            , MPI_DOUBLE, comm.up              , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_down.data()              , m_down.size()             , MPI_DOUBLE, comm.down            , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 5;
        MPI_Isend(m_frontout.data()          , m_frontout.size()         , MPI_DOUBLE, comm.front           , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_back.data()              , m_back.size()             , MPI_DOUBLE, comm.back            , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 6;
        MPI_Isend(m_backout.data()           , m_backout.size()          , MPI_DOUBLE, comm.back            , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_front.data()             , m_front.size()            , MPI_DOUBLE, comm.front           , mar, comm.comm, &mpi_requests_recv[mar-1]);
    // 12 lines
    mar = 7;
        MPI_Isend(m_frontleftout.data()      , m_frontleftout.size()     , MPI_DOUBLE, comm.frontleft       , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backright.data()         , m_backright.size()        , MPI_DOUBLE, comm.backright       , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 8;
        MPI_Isend(m_backrightout.data()      , m_backrightout.size()      , MPI_DOUBLE, comm.backright      , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontleft.data()         , m_frontleft.size()         , MPI_DOUBLE, comm.frontleft      , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 9;
        MPI_Isend(m_frontrightout.data()     , m_frontrightout.size()     , MPI_DOUBLE, comm.frontright     , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backleft.data()          , m_backleft.size()          , MPI_DOUBLE, comm.backleft       , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 10;
        MPI_Isend(m_backleftout.data()       , m_backleftout.size()       , MPI_DOUBLE, comm.backleft       , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontright.data()        , m_frontright.size()        , MPI_DOUBLE, comm.frontright     , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 11;
        MPI_Isend(m_leftupout.data()         , m_leftupout.size()         , MPI_DOUBLE, comm.leftup         , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_rightdown.data()         , m_rightdown.size()         , MPI_DOUBLE, comm.rightdown      , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 12;
        MPI_Isend(m_rightdownout.data()      , m_rightdownout.size()      , MPI_DOUBLE, comm.rightdown      , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_leftup.data()            , m_leftup.size()            , MPI_DOUBLE, comm.leftup         , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 13;
        MPI_Isend(m_leftdownout.data()       , m_leftdownout.size()       , MPI_DOUBLE, comm.leftdown       , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_rightup.data()           , m_rightup.size()           , MPI_DOUBLE, comm.rightup        , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 14;
        MPI_Isend(m_rightupout.data()        , m_rightupout.size()        , MPI_DOUBLE, comm.rightup        , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_leftdown.data()          , m_leftdown.size()          , MPI_DOUBLE, comm.leftdown       , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 15;
        MPI_Isend(m_frontupout.data()        , m_frontupout.size()        , MPI_DOUBLE, comm.frontup        , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backdown.data()          , m_backdown.size()          , MPI_DOUBLE, comm.backdown       , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 16;
        MPI_Isend(m_backdownout.data()       , m_backdownout.size()       , MPI_DOUBLE, comm.backdown       , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontup.data()           , m_frontup.size()           , MPI_DOUBLE, comm.frontup        , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 17;
        MPI_Isend(m_frontdownout.data()      , m_frontdownout.size()      , MPI_DOUBLE, comm.frontdown      , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backup.data()            , m_backup.size()            , MPI_DOUBLE, comm.backup         , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 18;
        MPI_Isend(m_backupout.data()         , m_backupout.size()         , MPI_DOUBLE, comm.backup         , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontdown.data()         , m_frontdown.size()         , MPI_DOUBLE, comm.frontdown      , mar, comm.comm, &mpi_requests_recv[mar-1]);
    // 8 points
    mar = 19;
        MPI_Isend(m_frontleftdownout.data()  , m_frontleftdownout.size()  , MPI_DOUBLE, comm.frontleftdown  , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backrightup.data()       , m_backrightup.size()       , MPI_DOUBLE, comm.backrightup    , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 20;
        MPI_Isend(m_backrightupout.data()    , m_backrightupout.size()    , MPI_DOUBLE, comm.backrightup    , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontleftdown.data()     , m_frontleftdown.size()     , MPI_DOUBLE, comm.frontleftdown  , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 21;
        MPI_Isend(m_backleftupout.data()     , m_backleftupout.size()     , MPI_DOUBLE, comm.backleftup     , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontrightdown.data()    , m_frontrightdown.size()    , MPI_DOUBLE, comm.frontrightdown , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 22;
        MPI_Isend(m_frontrightdownout.data() , m_frontrightdownout.size() , MPI_DOUBLE, comm.frontrightdown , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backleftup.data()        , m_backleftup.size()        , MPI_DOUBLE, comm.backleftup     , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 23;
        MPI_Isend(m_backleftdownout.data()   , m_backleftdownout.size()   , MPI_DOUBLE, comm.backleftdown   , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontrightup.data()      , m_frontrightup.size()      , MPI_DOUBLE, comm.frontrightup   , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 24;
        MPI_Isend(m_frontrightupout.data()   , m_frontrightupout.size()   , MPI_DOUBLE, comm.frontrightup   , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backleftdown.data()      , m_backleftdown.size()      , MPI_DOUBLE, comm.backleftdown   , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 25;
        MPI_Isend(m_frontleftupout.data()    , m_frontleftupout.size()    , MPI_DOUBLE, comm.frontleftup    , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_backrightdown.data()     , m_backrightdown.size()     , MPI_DOUBLE, comm.backrightdown  , mar, comm.comm, &mpi_requests_recv[mar-1]);

    mar = 26;
        MPI_Isend(m_backrightdownout.data()  , m_backrightdownout.size()  , MPI_DOUBLE, comm.backrightdown  , mar, comm.comm, &mpi_requests_send[mar-1]);

        MPI_Irecv(m_frontleftup.data()       , m_frontleftup.size()       , MPI_DOUBLE, comm.frontleftup    , mar, comm.comm, &mpi_requests_recv[mar-1]);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Waitall(26,mpi_requests_send,MPI_STATUS_IGNORE);
        MPI_Waitall(26,mpi_requests_recv,MPI_STATUS_IGNORE);
}

void LBM::unpack()
{
    // 6 faces

    Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), Kokkos::ALL, Kokkos::ALL), m_left);

    Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), Kokkos::ALL, Kokkos::ALL), m_right);

    Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(0, ghost)), m_down);

    Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(lz-ghost, lz)), m_up);

    Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, std::make_pair(0, ghost), Kokkos::ALL), m_front);

    Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, std::make_pair(ly-ghost, ly), Kokkos::ALL), m_back);
    // 12 lines

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz-ghost, lz)), m_leftup);

   
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz-ghost, lz)), m_rightup);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(0, ghost), std::make_pair(l_s[2], l_e[2])), m_frontleft);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(0, ghost), std::make_pair(l_s[2], l_e[2])), m_frontright);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(0, ghost)), m_leftdown);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(l_s[1], l_e[1]), std::make_pair(0, ghost)), m_rightdown);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(ly-ghost, ly), std::make_pair(l_s[2], l_e[2])), m_backleft);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(ly-ghost, ly), std::make_pair(l_s[2], l_e[2])), m_backright);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(0, ghost), std::make_pair(lz-ghost, lz)), m_frontup);

    
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(0, ghost), std::make_pair(0, ghost)), m_frontdown);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly-ghost, ly), std::make_pair(lz-ghost, lz)), m_backup);


        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly-ghost, ly), std::make_pair(0, ghost)), m_backdown);
    // 8 points

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(0, ghost), std::make_pair(0, ghost)), m_frontleftdown);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(0, ghost), std::make_pair(0, ghost)), m_frontrightdown);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(ly-ghost, ly), std::make_pair(0, ghost)), m_backleftdown);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(ly-ghost, ly), std::make_pair(0, ghost)), m_backrightdown);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(0, ghost), std::make_pair(lz-ghost, lz)), m_frontleftup);
  
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(0, ghost), std::make_pair(lz-ghost, lz)), m_frontrightup);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(0, ghost), std::make_pair(ly-ghost, ly), std::make_pair(lz-ghost, lz)), m_backleftup);

        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(lx-ghost, lx), std::make_pair(ly-ghost, ly), std::make_pair(lz-ghost, lz)), m_backrightup);
    
    Kokkos::fence();
};