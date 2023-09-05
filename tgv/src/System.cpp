#include "System.hpp"
#include "iostream"

using namespace std;
System::System(const int rank)
{
    // physical system definition
    this->cs2 = 1.0 / 3.0;
    this->cs = sqrt(cs2);

    fstream input;
    input.open("input.in");
    input >> this->rho0 >> this->R >> this->Re;
    input >> this->u0 >> this->Time >> this->inter;
    input >> this->sx >> this->sy >> this->sz;
    
    this->Time=100;  //(int)(this->Time*this->sx/3.1415926/this->u0/2);
    this->inter=1000;  //(int)(this->inter*this->sx/this->u0/3.1415926/2);
    this->Ma = this->u0 / this->cs;
    this->miu = this->rho0 * this->u0 * this->sx/2/3.1415926 / this->Re;
    this->tau = this->u0 * this->sx / Re / cs2/2/3.1415926;

    if (rank == 0) std::cout << "System Definition: Done" << std::endl;
}

void System::Monitor()
{

    std::cout << "============================" << std::endl
              << "TGV" << std::endl
              << "Re    =" << this->Re << std::endl
              << "Ma    =" << this->Ma << std::endl
              << "rho   =" << this->rho0 << std::endl
              << "miu   =" << this->miu << std::endl
              << "tau   =" << this->tau << std::endl
              << "Time  =" << this->Time << std::endl
              << "inter =" << this->inter << std::endl
              << "nx    =" << this->sx << std::endl
              << "ny    =" << this->sy << std::endl
              << "nz    =" << this->sz << std::endl
              << "============================" << std::endl;
};
