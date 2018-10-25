#ifndef NODES_H
#define NODES_H

#include "Data.h"
#include <Eigen/Core>
#include <Eigen/Dense>


class Nodes
{
    public:
        Nodes();
        //constant kernel
        Eigen::MatrixXd Nodes::constant(std::vector<double> vec);
        //squared exponential kernel
        Eigen::MatrixXd Nodes::squaredExponential(std::vector<double> vec);
        //periodic kernel
        Eigen::MatrixXd Nodes::periodic(std::vector<double> vec);
        //quasi periodic kernel
        Eigen::MatrixXd Nodes::quasiPeriodic(std::vector<double> vec);

    private:
        Eigen::MatrixXd C {Data::get_instance().N(), Data::get_instance().N()};
    
};

#endif // NODES_H
