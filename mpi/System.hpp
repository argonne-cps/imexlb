#ifndef _SYSTEM_H_
#define _SYSTEM_H_
#include "lbm.hpp"
#include <cmath>
#include <iostream>

class System
{

public:
    System();
    void Monitor();

    int sx, sy, sz;
    int Time, inter;
    double miu, u0, R, rho0, Ma, tau;

    int Re;
    double cs2, cs;
};
#endif
