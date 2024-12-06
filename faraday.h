#ifndef FARADAY_STRUCT
#define FARADAY_STRUCT

#include "Eigen/Dense"
#include "Eigen/Sparse"

struct Faraday {

    Eigen::MatrixXd P;
    Eigen::MatrixXd N;
	std::vector<std::vector<int>> my_cage_points;
	Eigen::MatrixXd bb;

    // tet stuff
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd TV;
	Eigen::MatrixXi TT;
	Eigen::MatrixXi TF;
    Eigen::VectorXi VM,FM;
    Eigen::MatrixXd H,R;
    Eigen::VectorXi TM,TR,PT;
    Eigen::MatrixXi FT,TN;
    Eigen::MatrixXd BC;
    int num_regions;

    Eigen::VectorXi is_cage_tv;
    Eigen::VectorXi is_bdry_tv;

    std::vector<std::vector<int>> my_tets;

    // numerical
    Eigen::SparseMatrix<double> L;
    std::unordered_map<int, int> global_to_matrix_ordering;

    Eigen::SparseMatrix<double> grad;
    Eigen::VectorXd vols;
    Eigen::MatrixXd u;
    Eigen::MatrixXd u_grad;
    Eigen::MatrixXd v_theta;
    Eigen::MatrixXd v_theta_grad;
    Eigen::VectorXd max;
    Eigen::MatrixXd max_grad;
    Eigen::MatrixXd max_grad_normalized;

    Eigen::MatrixXd N_est;

};

#endif