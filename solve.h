#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "faraday.h"
#include "pc.h"
#include <igl/slice.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>

#include "gurobi_c++.h"

#include "geometrycentral/numerical/linear_solvers.h"

Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::VectorXd &func);

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::VectorXd &func);

void build_f_to_v_matrix(struct Faraday &f);

Eigen::VectorXd solvePotentialOverDirs_Gurobi(struct Faraday &f);

void solvePotentialOverDirs(struct Faraday &f);
void solvePotentialPointCharges(struct Faraday &f, std::vector<int> &pt_constraints);

void solveMaxFunction(struct Faraday &f);

void estimateNormals(struct Faraday &f);

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, Eigen::VectorXd &bdry_vals);
Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, std::vector<int> &pt_constraints, double const_val);

Eigen::SparseMatrix<double> computeFaraday(struct Faraday &f);
std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f, std::vector<int> &pt_constraints);

// Eigen::VectorXd solveDirichletProblem(struct Faraday &f, Eigen::VectorXd &bdry_vals);