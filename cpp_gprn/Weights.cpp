#include "Weights.h"
#include "Data.h"
#include <iostream>
#include <fstream>

using namespace std;

Weights::Weights()
{

}

//time
const vector<double>& t = Data::get_instance().get_t();

double Weights::constant( std::vector<double> vec )
{
    double kernel = vec[0];
    return kernel;

}


