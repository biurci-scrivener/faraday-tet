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

    // octree & knn stuff
    std::vector<std::vector<int>> PI;
    Eigen::MatrixXi CH;
    Eigen::MatrixXd CN;
    Eigen::VectorXd W;
    Eigen::MatrixXi knn;

    Eigen::VectorXd cage_radii;

    Eigen::VectorXi is_cage_tv;
    Eigen::VectorXi is_bdry_tv;

    std::vector<std::vector<int>> my_tets;

    // numerical
    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> D_inv;
    Eigen::SparseMatrix<double> L;
    std::unordered_map<int, int> global_to_matrix_ordering;

    Eigen::SparseMatrix<double> grad;
    Eigen::SparseMatrix<double> f_to_v; // turns a function on faces (tets.) into a function on verts.
    Eigen::VectorXd vols;
    Eigen::MatrixXd u;
    Eigen::MatrixXd u_grad;
    Eigen::MatrixXd v_theta;
    Eigen::MatrixXd v_theta_grad;
    Eigen::VectorXd max;
    Eigen::MatrixXd max_grad;
    Eigen::VectorXd max_var;
    Eigen::MatrixXd max_grad_normalized;

    Eigen::MatrixXd N_est;

};

#endif