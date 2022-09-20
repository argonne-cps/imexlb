#include "System.hpp"
#include "iostream"

using namespace std;
System::System()
{
    // system defination
        this->cs2 = 1.0 / 3.0;
    this->cs = sqrt(cs2);

    fstream input;
    input.open("input.in");
    input >> this->rho0 >> this->R >> this->Re;
    input >> this->u0 >> this->Time >> this->inter;
    input >> this->sx >> this->sy >> this->sz;

    this->Ma = this->u0 / this->cs;
    this->miu = this->rho0 * this->u0 * this->R / this->Re;
    this->tau = this->u0 * this->R / Re / cs2;

    this->Time = pow(2 * R, 2) / this->miu;
    this->inter = 1; //this->Time / 10;
}

void System::Monitor()
{

    std::cout << "============================" << std::endl
              << "3D Cylinder Flow" << std::endl
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
