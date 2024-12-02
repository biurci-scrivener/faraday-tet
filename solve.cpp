#include "solve.h"

#define USE_BILAPLACIAN false

Eigen::MatrixXd potential_dirs = ico_pts;

Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::VectorXd &func) {
    Eigen::VectorXd res = f.grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    return g_f;
}

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::VectorXd &func) {
    Eigen::VectorXd res = f.grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    Eigen::MatrixXd g_tv = Eigen::MatrixXd::Zero(f.TV.rows(), 3);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        double w_sum = 0.;
        for (int tet: f.my_tets[i]) {
            g_tv.row(i) += g_f.row(tet) * f.vols[tet];
            w_sum += f.vols[tet];
        }
        if (f.my_tets[i].size() > 0) g_tv.row(i) /= w_sum;
    }
    return g_tv;
}

Eigen::VectorXd solvePotentialOverDirs_Gurobi(struct Faraday &f) {

    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> D;
    Eigen::SparseMatrix<double> D_inv;

    std::cout << "Building Laclacian" << std::endl;
    igl::cotmatrix(f.TV, f.TT, M);
    std::cout << "\tDone" << std::endl;
    std::cout << "Building mass matrix" << std::endl;
    igl::massmatrix(f.TV, f.TT, igl::MASSMATRIX_TYPE_BARYCENTRIC, D);
    std::cout << "\tDone" << std::endl;
    igl::invert_diag(D, D_inv);

    Eigen::SparseMatrix<double> L = D_inv * M;

    // generate potential in absence of shielding

    Eigen::VectorXd boundary_vals(f.TV.rows());
    Eigen::VectorXd dir = potential_dirs.row(0);
    for (int j = 0; j < f.TV.rows(); j++) boundary_vals[j] = f.TV.row(j).dot(dir);

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());

    try {
        GRBEnv env = GRBEnv();

        GRBModel model = GRBModel(env);

        // Create variables

        std::cout << "Creating decision variables" << std::endl;
        
        std::vector<GRBVar> u;
        for (size_t i = 0; i < f.TV.rows(); i++) {
            u.push_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS));
        }

        // Add constraints

        std::cout << "Adding constraints" << std::endl;

        for (size_t i = 0; i < f.TV.rows(); i++) {
            if (f.is_bdry_tv(i)) {
                model.addConstr(u[i] == boundary_vals[i]);
                std::cout << "Added boundary constraint: " << i << std::endl;
            } else if (f.is_cage_tv(i)) {
                size_t next_cage = i + 1;
                while ((next_cage < f.TV.rows()) && (!f.is_cage_tv(next_cage))) next_cage++;
                if (next_cage != f.TV.rows()) {
                    model.addConstr(u[i] == u[next_cage]);
                    std::cout << "Added cage constraint: " << i << " " << next_cage << std::endl;
                }
            }
        }
        

        // Set objective

        std::cout << "Setting objective" << std::endl;

        GRBQuadExpr obj = 0.0;

        for (int k=0; k<L.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
                obj += 0.5 * it.value() * u[it.row()] * u[it.col()];
            }
        }
        
        model.setObjective(obj, GRB_MINIMIZE);

        model.optimize();

        for (size_t i = 0; i < f.TV.rows(); i++) {
            sol[i] = u[i].get(GRB_DoubleAttr_X);
        }

    } catch(GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
        exit(-1);
    } catch(...) {
        std::cout << "Exception during optimization" << std::endl;
        exit(-1);
    }
    
    return sol;
}

void solvePotentialOverDirs(struct Faraday &f, std::vector<int> &pt_constraints) {
    

    std::cout << "Starting solve for potential over all directions." << std::endl;

    f.u = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows());
    f.u_grad = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows() * 3);
    f.v_theta = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows());
    f.v_theta_grad = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows() * 3);

    // pre-solve for Faraday effect

    std::unordered_map<int, int> global_to_matrix_ordering;
    Eigen::SparseMatrix<double> KKT;
    std::tie(global_to_matrix_ordering, KKT) = computeFaraday(f, pt_constraints);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver(KKT);
    std::cout << "Initialized solver" << std::endl;

    // pre-solve for potential sans shielding

    std::unordered_map<int, int> global_to_matrix_ordering_base;
    Eigen::SparseMatrix<double> KKT_base;
    std::tie(global_to_matrix_ordering_base, KKT_base) = computeBasePotential(f, pt_constraints);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver_base(KKT_base);
    std::cout << "Initialized solver" << std::endl;

    /*

        Let u, v_theta be the potentials with and without shielding
        For each field direction, compute
            - u
            - v_theta
            - gradients of each (on vertices)

    */

    for (int i = 0; i < potential_dirs.rows(); i++) {
        std::cout << "\tSolving for directions " << i << std::endl;

        Eigen::VectorXd boundary_vals(f.TV.rows());
        Eigen::VectorXd dir = potential_dirs.row(i);
        for (int j = 0; j < f.TV.rows(); j++) boundary_vals[j] = f.TV.row(j).dot(dir);
        
        // f.v_theta.col(i) = boundary_vals;
        Eigen::VectorXd res_base = solveBasePotential(f, solver_base, global_to_matrix_ordering_base, boundary_vals, pt_constraints);
        f.v_theta.col(i) = res_base;
        Eigen::MatrixXd boundary_vals_grad = grad_tv(f, res_base);

        f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
        f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
        f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

        Eigen::VectorXd res = solveFaraday(f, solver, global_to_matrix_ordering, boundary_vals, pt_constraints);
        f.u.col(i) = res;
        Eigen::MatrixXd res_grad = grad_tv(f, res);
        f.u_grad.col(i * 3) = res_grad.col(0);
        f.u_grad.col(i * 3 + 1) = res_grad.col(1);
        f.u_grad.col(i * 3 + 2) = res_grad.col(2);
    }

}

void solveFieldDifference(struct Faraday &f) {

    /*

        Let u, v_theta be the potentials with and without shielding
        For each field direction, compute |grad_u - grad_v_theta|
            (Note: these gradients live on vertices)

        Then, take the maximum value (of this norm) across all directions
        Finally, take gradient (on vertices)

    */

    Eigen::MatrixXd gradmag = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows());
    f.max = Eigen::VectorXd::Zero(f.TV.rows());
    f.max_grad = Eigen::MatrixXd::Zero(f.TV.rows(), 3);

    for (int i = 0; i < potential_dirs.rows(); i++) {
        std::cout << i << std::endl;
        Eigen::VectorXd grad_diff_norm = (f.u_grad.middleCols(i * 3, 3) - f.v_theta_grad.middleCols(i * 3, 3)).rowwise().norm();
        gradmag.col(i) = grad_diff_norm;
    }

    f.max = gradmag.rowwise().maxCoeff();
    f.max_grad = grad_tv(f, f.max);
    f.max_grad_normalized = f.max_grad.rowwise().normalized();

}

void estimateNormals(struct Faraday &f) {
    
    /*
    
        Normals are estimated from max_grad

    */

    f.N_est = Eigen::MatrixXd::Zero(f.P.rows(), 3);
    std::cout << f.max_grad.rows() << std::endl;
    std::cout << f.max_grad.cols() << std::endl;
    
    for (int i = 0; i < f.P.rows(); i++) {
        for (int cage: f.my_cage_points[i]) {
            f.N_est.row(i) += f.max_grad.row(cage);
        }
        f.N_est.row(i) /= f.my_cage_points[i].size();
        f.N_est.row(i) = f.N_est.row(i).normalized();
    }

}

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, Eigen::VectorXd &bdry_vals, std::vector<int> &pt_constraints) {

    int constraints_size = f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt_constraints.size();

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(f.TV.rows() + constraints_size);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            int current_idx = global_to_matrix_ordering[i];
            RHS[f.TV.rows() + current_idx] = bdry_vals[i];
        }
    }

    double pt_constraint_val = bdry_vals.maxCoeff();

    for (size_t i = 0; i < pt_constraints.size(); i++) {
        RHS[f.TV.rows() + f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + i] = pt_constraint_val;
    }

    Eigen::VectorXd u = solver.solve(RHS);
    // if (solver.info() != Eigen::Success) {
    //     std::cout << "ERROR: Solve failed!" << std::endl;
    //     exit(-1);
    // }

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());

    for (size_t i = 0; i < f.TV.rows(); i++) sol[i] = u[global_to_matrix_ordering[i]];

    return sol;
}

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f, std::vector<int> &pt_constraints) {

    // bdry_vals should be TV.rows() long, but only the entries for boundary vertices will be considered
    /*
        reindex all leaf cells so that they're neatly ordered as follows
        boundary, cage, interior
    */ 

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_cage_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if ((!f.is_bdry_tv(i)) && (!f.is_cage_tv(i))) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    if (matrix_count != f.TV.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    int constraints_size = f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt_constraints.size();
    Eigen::SparseMatrix<double> KKT(f.TV.rows() + constraints_size, f.TV.rows() + constraints_size);
    std::vector<Eigen::Triplet<double>> triplets;

    // build Laplacian 
    // this is the Laplacian for ALL verts

    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> D;
    Eigen::SparseMatrix<double> D_inv;

    igl::cotmatrix(f.TV, f.TT, M);
    igl::massmatrix(f.TV, f.TT, igl::MASSMATRIX_TYPE_BARYCENTRIC, D);
    igl::invert_diag(D, D_inv);

    Eigen::SparseMatrix<double> L = D_inv * M;
    if (USE_BILAPLACIAN) L = L * L;

    for (int k=0; k<L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
            triplets.push_back(Eigen::Triplet<double>(global_to_matrix_ordering[it.row()], global_to_matrix_ordering[(it.col())], it.value()));
        }
    }

    std::cout << "\tAdded Laplacian triplets" << std::endl;

    for (int bdry = 0; bdry < f.is_bdry_tv.sum(); bdry++) {
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + bdry, bdry, 1.));
        triplets.push_back(Eigen::Triplet<double>(bdry, f.TV.rows() + bdry, 1.));
        // std::cout << f.TV.rows() + bdry << " " << bdry << " " << 1. << "\n";
    }
    for (int cage = f.is_bdry_tv.sum(); cage < f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1; cage++) {
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + cage, cage, 1.));
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + cage, cage + 1, -1.));
        triplets.push_back(Eigen::Triplet<double>(cage, f.TV.rows() + cage, 1.));
        triplets.push_back(Eigen::Triplet<double>(cage + 1, f.TV.rows() + cage, -1.));
        // std::cout << f.TV.rows() + cage << " " << cage << " " << 1. << "\n";
        // std::cout << f.TV.rows() + cage << " " << cage + 1 << " " << -1. << "\n";
    }
    for (int pt = 0; pt < pt_constraints.size(); pt++) {
        int pt_idx = global_to_matrix_ordering[pt_constraints[pt]];
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt, pt_idx, 1.));
        triplets.push_back(Eigen::Triplet<double>(pt_idx, f.TV.rows() + f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt, 1.));
    }

    std::cout << "\tAdded constraint triplets. About to set KKT matrix" << std::endl;

    KKT.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tSet KKT matrix. Size is " << KKT.rows() << " x " << KKT.cols() << ". Decomposing, this may a take a while..." << std::endl;

    return std::make_tuple(global_to_matrix_ordering, KKT);

}

Eigen::VectorXd solveBasePotential(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, Eigen::VectorXd &bdry_vals, std::vector<int> &pt_constraints) {

    int constraints_size = f.is_bdry_tv.sum() + pt_constraints.size();

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(f.TV.rows() + constraints_size);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            int current_idx = global_to_matrix_ordering[i];
            RHS[f.TV.rows() + current_idx] = bdry_vals[i];
        }
    }

    double pt_constraint_val = bdry_vals.maxCoeff();

    for (size_t i = 0; i < pt_constraints.size(); i++) {
        RHS[f.TV.rows() + f.is_bdry_tv.sum() + i] = pt_constraint_val;
    }

    Eigen::VectorXd u = solver.solve(RHS);
    // if (solver.info() != Eigen::Success) {
    //     std::cout << "ERROR: Solve failed!" << std::endl;
    //     exit(-1);    
    // }

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());

    for (size_t i = 0; i < f.TV.rows(); i++) sol[i] = u[global_to_matrix_ordering[i]];

    return sol;

}

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeBasePotential(struct Faraday &f, std::vector<int> &pt_constraints) {

    // bdry_vals should be TV.rows() long, but only the entries for boundary vertices will be considered

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (!f.is_bdry_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    if (matrix_count != f.TV.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    int constraints_size = f.is_bdry_tv.sum() + pt_constraints.size();
    Eigen::SparseMatrix<double> KKT(f.TV.rows() + constraints_size, f.TV.rows() + constraints_size);
    std::vector<Eigen::Triplet<double>> triplets;

    // build Laplacian 
    // this is the Laplacian for ALL verts

    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> D;
    Eigen::SparseMatrix<double> D_inv;

    igl::cotmatrix(f.TV, f.TT, M);
    igl::massmatrix(f.TV, f.TT, igl::MASSMATRIX_TYPE_BARYCENTRIC, D);
    igl::invert_diag(D, D_inv);

    Eigen::SparseMatrix<double> L = D_inv * M;
    if (USE_BILAPLACIAN) L = L * L;

    for (int k=0; k<L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
            triplets.push_back(Eigen::Triplet<double>(global_to_matrix_ordering[it.row()], global_to_matrix_ordering[(it.col())], it.value()));
        }
    }

    std::cout << "\tAdded Laplacian triplets" << std::endl;

    for (int bdry = 0; bdry < f.is_bdry_tv.sum(); bdry++) {
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + bdry, bdry, 1.));
        triplets.push_back(Eigen::Triplet<double>(bdry, f.TV.rows() + bdry, 1.));
    }
    for (int pt = 0; pt < pt_constraints.size(); pt++) {
        int pt_idx = global_to_matrix_ordering[pt_constraints[pt]];
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + f.is_bdry_tv.sum() + pt, pt_idx, 1.));
        triplets.push_back(Eigen::Triplet<double>(pt_idx, f.TV.rows() + f.is_bdry_tv.sum() + pt, 1.));
    }

    std::cout << "\tAdded constraint triplets. About to set KKT matrix" << std::endl;

    KKT.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tSet KKT matrix. Size is " << KKT.rows() << " x " << KKT.cols() << ". Decomposing, this may a take a while..." << std::endl;

    return std::make_tuple(global_to_matrix_ordering, KKT);

}