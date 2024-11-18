#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "faraday.h"
#include <igl/slice.h>
#include <igl/cotmatrix.h>

Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::SparseMatrix<double> &grad, Eigen::VectorXd &func);

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::SparseMatrix<double> &grad, Eigen::VectorXd &func);

Eigen::VectorXd solveFaraday(struct Faraday &f, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, Eigen::VectorXd &bdry_vals);

std::unordered_map<int, int> computeFaraday(struct Faraday &f, Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> &solver);

// Eigen::VectorXd solveDirichletProblem(struct Faraday &f, Eigen::VectorXd &bdry_vals);