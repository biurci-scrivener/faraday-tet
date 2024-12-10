#include "solve.h"

Eigen::MatrixXd potential_dirs = ico_pts;

Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::VectorXd &func) {
    Eigen::VectorXd res = f.grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    return g_f;
}

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::VectorXd &func) {
    Eigen::VectorXd res = f.grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    return f.f_to_v * g_f;
}

void build_f_to_v_matrix(struct Faraday &f) {

    f.f_to_v = Eigen::SparseMatrix<double>(f.TV.rows(), f.TT.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < f.TV.rows(); i++) {
        double w_sum = 0;
        for (int tet: f.my_tets[i]) {
            w_sum += f.vols[tet];
        }
        for (int tet: f.my_tets[i]) {
            triplets.push_back(Eigen::Triplet<double>(i, tet, f.vols[tet] / w_sum));
        }
    }
    f.f_to_v.setFromTriplets(triplets.begin(), triplets.end());
}

void solvePotentialOverDirs(struct Faraday &f) {
    

    std::cout << "Starting solve for potential over all directions." << std::endl;
    size_t no_fields = potential_dirs.rows();

    f.u = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields);
    f.u_grad = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields * 3);
    f.v_theta = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields);
    f.v_theta_grad = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields * 3);

    // pre-solve for Faraday effect, linear fields

    Eigen::SparseMatrix<double> LHS = computeFaraday(f);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver(LHS);
    std::cout << "Initialized solver" << std::endl;
    
    /*

        Let u, v_theta be the potentials with and without shielding
        For each field direction, compute
            - u
            - v_theta
            - gradients of each (on vertices)

    */

    for (int i = 0; i < potential_dirs.rows(); i++) {
        std::cout << "\tSolving for direction " << i << std::endl;

        Eigen::VectorXd boundary_vals(f.TV.rows());
        Eigen::VectorXd dir = potential_dirs.row(i);
        for (int j = 0; j < f.TV.rows(); j++) boundary_vals[j] = f.TV.row(j).dot(dir);
        f.v_theta.col(i) = boundary_vals;  
        std::cout << "\t\tComputed boundary values" << std::endl;

        // Eigen::MatrixXd boundary_vals_grad = grad_tv(f, boundary_vals);
        // std::cout << "\t\tComputed grad. of bdry. values" << std::endl;
        // f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
        // f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
        // f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

        Eigen::VectorXd res = solveFaraday(f, solver, boundary_vals);
        std::cout << "\t\tSolved for u" << std::endl;
        f.u.col(i) = res;
        Eigen::MatrixXd res_grad = grad_tv(f, res);
        std::cout << "\t\tComputed grad. of u" << std::endl;
        f.u_grad.col(i * 3) = res_grad.col(0);
        f.u_grad.col(i * 3 + 1) = res_grad.col(1);
        f.u_grad.col(i * 3 + 2) = res_grad.col(2);
    }

}

void solvePotentialPointCharges(struct Faraday &f, std::vector<int> &pt_constraints) {

    if (!pt_constraints.size()) {
        f.u.conservativeResize(f.u.rows(), potential_dirs.rows());
        f.u_grad.conservativeResize(f.u_grad.rows(), (potential_dirs.rows() * 3));
        f.v_theta.conservativeResize(f.v_theta.rows(), potential_dirs.rows());
        f.v_theta_grad.conservativeResize(f.v_theta_grad.rows(), (potential_dirs.rows() * 3));
        std::cout << "\tNo point constaints specified, returning." << std::endl;
        return;
    }

    std::cout << "\tSolving for point charge field" << std::endl;

    f.u.conservativeResize(f.u.rows(), potential_dirs.rows() + 1);
    f.u_grad.conservativeResize(f.u_grad.rows(), (potential_dirs.rows() * 3) + 3);
    f.v_theta.conservativeResize(f.v_theta.rows(), potential_dirs.rows() + 1);
    f.v_theta_grad.conservativeResize(f.v_theta_grad.rows(), (potential_dirs.rows() * 3) + 3);

    // pre-solve for Faraday effect, point charges

    std::unordered_map<int, int> global_to_matrix_ordering_pt_charge;
    Eigen::SparseMatrix<double> LHS;
    std::tie(global_to_matrix_ordering_pt_charge, LHS) = computeFaraday(f, pt_constraints);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver_pt_charge(LHS);
    std::cout << "Initialized solver" << std::endl;

    // solve

    int i = potential_dirs.rows();

    double pt_const_val = f.v_theta.col(0).maxCoeff();
    Eigen::VectorXd pt_base = Eigen::VectorXd::Zero(f.TV.rows());
    f.v_theta.col(i) = pt_base;

    Eigen::MatrixXd boundary_vals_grad = grad_tv(f, pt_base);
    f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
    f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
    f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

    Eigen::VectorXd res = solveFaraday(f, solver_pt_charge, global_to_matrix_ordering_pt_charge, pt_constraints, pt_const_val);
    f.u.col(i) = res;
    Eigen::MatrixXd res_grad = grad_tv(f, res);
    f.u_grad.col(i * 3) = res_grad.col(0);
    f.u_grad.col(i * 3 + 1) = res_grad.col(1);
    f.u_grad.col(i * 3 + 2) = res_grad.col(2);
    
}

void solveMaxFunction(struct Faraday &f) {

    /*

        Let u be the potential with shielding (for a given direction)
        For each field direction, compute |grad_u|
            (Note: these gradients live on vertices)

        Then, take the maximum value (of this norm) across all directions
        Finally, take gradient (on vertices)

    */

    std::cout << "Computing max. grad." << std::endl;

    Eigen::MatrixXd gradmag = Eigen::MatrixXd::Zero(f.TV.rows(), f.u.cols());
    f.max = Eigen::VectorXd::Zero(f.TV.rows());
    f.max_var = Eigen::VectorXd::Zero(f.TV.rows());
    f.max_grad = Eigen::MatrixXd::Zero(f.TV.rows(), 3);

    for (int i = 0; i < f.u.cols(); i++) {
        Eigen::VectorXd grad_norm = (f.u_grad.middleCols(i * 3, 3)).rowwise().norm();
        gradmag.col(i) = grad_norm;
    }

    f.max = gradmag.rowwise().maxCoeff();
    // compute variance
    Eigen::VectorXd max_mean = gradmag.rowwise().mean();
    f.max_var = Eigen::VectorXd::Zero(f.TV.rows());
    for (int i = 0; i < gradmag.cols(); i++) {
        // check this, i think it's wrong
        Eigen::VectorXd diff = (gradmag.col(i) - max_mean).array().pow(2) / ((double)gradmag.cols() - 1);
        f.max_var += diff;
    }

    f.max_grad = grad_tv(f, f.max);
    f.max_grad_normalized = f.max_grad.rowwise().normalized();

}

void estimateNormals(struct Faraday &f) {
    
    /*
    
        Normals are estimated from max_grad

    */

    std::cout << "Estimating normals" << std::endl;

    f.N_est = Eigen::MatrixXd::Zero(f.P.rows(), 3);
    
    for (int i = 0; i < f.P.rows(); i++) {
        for (int cage: f.my_cage_points[i]) {
            f.N_est.row(i) += f.max_grad.row(cage);
        }
        f.N_est.row(i) /= f.my_cage_points[i].size();
        f.N_est.row(i) = f.N_est.row(i).normalized();
    }

}

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, Eigen::VectorXd &bdry_vals) {

    int boundary_count = f.is_bdry_tv.sum();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(interior_count + 1);
    std::cout << "\t\tBuilding RHS" << std::endl;

    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {

            // this is just so that I can do an "is_interior" type call
            // using the interior_count thing from computeFaraday
            int i = f.global_to_matrix_ordering[it.row()];

            if ((i < interior_count) && (f.is_bdry_tv(it.col()))) {
                RHS[i] -= it.value() * bdry_vals(it.col());
            }

        }
    }

    std::cout << "\t\tSolving" << std::endl;
    Eigen::VectorXd u = solver.solve(RHS);
    std::cout << "\t\tSolved" << std::endl;

    // build solution vector
    // the value for the cage vertices is wrong.
    // let us fix it disgracefully (or at least figure out what it should be)

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            sol[i] = bdry_vals[i];
        } else if (f.is_cage_tv(i)) {
            sol[i] = u[interior_count];
        } else {
            sol[i] = u[f.global_to_matrix_ordering[i]];
        }
    }

    return sol;
}

Eigen::SparseMatrix<double> computeFaraday(struct Faraday &f) {

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    int boundary_count = f.is_bdry_tv.sum();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    for (size_t i = 0; i < f.TV.rows(); i++) {
        // "normal" interior vertices
        if ((!f.is_bdry_tv(i)) && (!f.is_cage_tv(i))) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // boundary vertices
        if (f.is_bdry_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // cage vertices
        if (f.is_cage_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    
    if (matrix_count != f.TV.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    Eigen::SparseMatrix<double> LHS(interior_count + 1, interior_count + 1);
    std::vector<Eigen::Triplet<double>> triplets;

    // split Laplacian into parts
    Eigen::SparseMatrix<double> L_NC(interior_count, interior_count);
    Eigen::SparseMatrix<double> L_CC(cage_count, cage_count);
    Eigen::SparseMatrix<double> L_NCC(interior_count, cage_count);
    Eigen::SparseMatrix<double> L_CNC(cage_count, interior_count);
    // std::vector<Eigen::Triplet<double>> L_NC_triplets;
    std::vector<Eigen::Triplet<double>> L_CC_triplets;
    std::vector<Eigen::Triplet<double>> L_NCC_triplets;
    std::vector<Eigen::Triplet<double>> L_CNC_triplets;

    std::cout << "Setting triplets" << std::endl;

    int offset = interior_count + boundary_count;
    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {
            int i = global_to_matrix_ordering[it.row()];
            int j = global_to_matrix_ordering[it.col()];
            if ((i < interior_count) && (f.is_cage_tv(it.col()))) {
                L_NCC_triplets.push_back(Eigen::Triplet<double>(i, j - offset, it.value()));
            } else if ((j < interior_count) && (f.is_cage_tv(it.row()))) {
                L_CNC_triplets.push_back(Eigen::Triplet<double>(i - offset, j, it.value()));
            } else if ((i < interior_count) && (j < interior_count)) {
                triplets.push_back(Eigen::Triplet<double>(i, j, it.value()));
            } else if ((f.is_cage_tv(it.row())) && (f.is_cage_tv(it.col()))) {
                L_CC_triplets.push_back(Eigen::Triplet<double>(i - offset, j - offset, it.value()));
            }
        }
    }

    L_CC.setFromTriplets(L_CC_triplets.begin(), L_CC_triplets.end());
    L_NCC.setFromTriplets(L_NCC_triplets.begin(), L_NCC_triplets.end());
    L_CNC.setFromTriplets(L_CNC_triplets.begin(), L_CNC_triplets.end());

    Eigen::VectorXd col_r = L_NCC * Eigen::VectorXd::Ones(cage_count);
    Eigen::VectorXd row_b = Eigen::RowVectorXd::Ones(cage_count) * L_CNC;
    double val_br = Eigen::RowVectorXd::Ones(cage_count) * L_CC * Eigen::VectorXd::Ones(cage_count);

    for (int i = 0; i < interior_count; i++) {
        triplets.push_back(Eigen::Triplet<double>(interior_count, i, row_b(i)));
        triplets.push_back(Eigen::Triplet<double>(i, interior_count, col_r(i)));
    }
    triplets.push_back(Eigen::Triplet<double>(interior_count, interior_count, val_br));

    LHS.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tDone" << std::endl;

    f.global_to_matrix_ordering = global_to_matrix_ordering;
    return LHS;

}

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, std::vector<int> &pt_constraints, double const_val) {

    // version for point constraints

    int boundary_count = f.is_bdry_tv.sum() + pt_constraints.size();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(interior_count + 1);
    std::cout << "\t\tBuilding RHS" << std::endl;

    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {

            // this is just so that I can do an "is_interior" type call
            // using the interior_count thing from computeFaraday
            int i = global_to_matrix_ordering[it.row()];
            
            if ((i < interior_count) && (std::find(pt_constraints.begin(), pt_constraints.end(), it.col()) != pt_constraints.end())) {
                RHS[i] -= it.value() * const_val;
            }

        }
    }

    std::cout << "\tSolving" << std::endl;
    Eigen::VectorXd u = solver.solve(RHS);
    std::cout << "\tSolved" << std::endl;

    // build solution vector

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (std::find(pt_constraints.begin(), pt_constraints.end(), i) != pt_constraints.end()) {
            sol[i] = const_val;
        } else if (f.is_cage_tv(i)) {
            sol[i] = u[interior_count];
        } else if (!f.is_bdry_tv(i)){
            sol[i] = u[global_to_matrix_ordering[i]];
        }
    }

    std::cout << "\tBuilt solution vector" << std::endl;

    return sol;
    
}

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f, std::vector<int> &pt_constraints) {

    // version for point constraints

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    int boundary_count = f.is_bdry_tv.sum() + pt_constraints.size();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    for (size_t i = 0; i < f.TV.rows(); i++) {
        // "normal" interior vertices
        if ((!f.is_bdry_tv(i)) && (!f.is_cage_tv(i)) && (std::find(pt_constraints.begin(), pt_constraints.end(), i) == pt_constraints.end())) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // boundary vertices
        if (f.is_bdry_tv(i) || (std::find(pt_constraints.begin(), pt_constraints.end(), i) != pt_constraints.end())) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // cage vertices
        if (f.is_cage_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    
    if (matrix_count != f.TV.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    Eigen::SparseMatrix<double> LHS(interior_count + 1, interior_count + 1);
    std::vector<Eigen::Triplet<double>> triplets;

    // split Laplacian into parts
    Eigen::SparseMatrix<double> L_NC(interior_count, interior_count);
    Eigen::SparseMatrix<double> L_CC(cage_count, cage_count);
    Eigen::SparseMatrix<double> L_NCC(interior_count, cage_count);
    Eigen::SparseMatrix<double> L_CNC(cage_count, interior_count);
    // std::vector<Eigen::Triplet<double>> L_NC_triplets;
    std::vector<Eigen::Triplet<double>> L_CC_triplets;
    std::vector<Eigen::Triplet<double>> L_NCC_triplets;
    std::vector<Eigen::Triplet<double>> L_CNC_triplets;

    std::cout << "Setting triplets" << std::endl;

    int offset = interior_count + boundary_count;
    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {
            int i = global_to_matrix_ordering[it.row()];
            int j = global_to_matrix_ordering[it.col()];
            if ((i < interior_count) && (f.is_cage_tv(it.col()))) {
                L_NCC_triplets.push_back(Eigen::Triplet<double>(i, j - offset, it.value()));
            } else if ((j < interior_count) && (f.is_cage_tv(it.row()))) {
                L_CNC_triplets.push_back(Eigen::Triplet<double>(i - offset, j, it.value()));
            } else if ((i < interior_count) && (j < interior_count)) {
                triplets.push_back(Eigen::Triplet<double>(i, j, it.value()));
            } else if ((f.is_cage_tv(it.row())) && (f.is_cage_tv(it.col()))) {
                L_CC_triplets.push_back(Eigen::Triplet<double>(i - offset, j - offset, it.value()));
            }
        }
    }

    L_CC.setFromTriplets(L_CC_triplets.begin(), L_CC_triplets.end());
    L_NCC.setFromTriplets(L_NCC_triplets.begin(), L_NCC_triplets.end());
    L_CNC.setFromTriplets(L_CNC_triplets.begin(), L_CNC_triplets.end());

    Eigen::VectorXd col_r = L_NCC * Eigen::VectorXd::Ones(cage_count);
    Eigen::VectorXd row_b = Eigen::RowVectorXd::Ones(cage_count) * L_CNC;
    double val_br = Eigen::RowVectorXd::Ones(cage_count) * L_CC * Eigen::VectorXd::Ones(cage_count);

    for (int i = 0; i < interior_count; i++) {
        triplets.push_back(Eigen::Triplet<double>(interior_count, i, row_b(i)));
        triplets.push_back(Eigen::Triplet<double>(i, interior_count, col_r(i)));
    }
    triplets.push_back(Eigen::Triplet<double>(interior_count, interior_count, val_br));

    LHS.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tDone" << std::endl;

    return std::make_tuple(global_to_matrix_ordering, LHS);

}