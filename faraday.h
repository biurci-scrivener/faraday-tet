#ifndef FARADAY_STRUCT
#define FARADAY_STRUCT

#include "Eigen/Dense"
#include "Eigen/Sparse"

struct Faraday {

    Eigen::MatrixXd P;
    Eigen::MatrixXd N;
    // Eigen::VectorXi is_boundary_point; // not needed. can just check if in first 8
	Eigen::VectorXi is_cage_point;
    Eigen::VectorXi is_cage_tv;
    Eigen::VectorXi is_bdry_tv;
	std::vector<std::vector<int>> my_cage_points;
    std::vector<std::vector<int>> my_tets;
	Eigen::MatrixXd bb;
    Eigen::MatrixXd TV;
	Eigen::MatrixXi TT;
	Eigen::MatrixXi TF;
    Eigen::MatrixXd BC;

};

#endif