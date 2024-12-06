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

void solvePotentialOverDirs(struct Faraday &f, bool use_bilaplacian) {
    

    std::cout << "Starting solve for potential over all directions." << std::endl;
    if (use_bilaplacian) {
        std::cout << "Using Bilaplacian" << std::endl;
    } else {
        std::cout << "Using Laplacian" << std::endl;
    }
    size_t no_fields = potential_dirs.rows();

    f.u = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields);
    f.u_grad = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields * 3);
    f.v_theta = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields);
    f.v_theta_grad = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields * 3);

    // pre-solve for Faraday effect, linear fields

    Eigen::SparseMatrix<double> KKT = computeFaraday(f, use_bilaplacian);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver(KKT);
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

        Eigen::MatrixXd boundary_vals_grad = grad_tv(f, boundary_vals);
        f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
        f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
        f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

        Eigen::VectorXd res = solveFaraday(f, solver, boundary_vals);
        f.u.col(i) = res;
        Eigen::MatrixXd res_grad = grad_tv(f, res);
        f.u_grad.col(i * 3) = res_grad.col(0);
        f.u_grad.col(i * 3 + 1) = res_grad.col(1);
        f.u_grad.col(i * 3 + 2) = res_grad.col(2);
    }

}

void solvePotentialPointCharges(struct Faraday &f, std::vector<int> &pt_constraints, bool use_bilaplacian) {

    std::cout << "\tSolving for point charge field" << std::endl;

    if (use_bilaplacian) {
        std::cout << "Using Bilaplacian" << std::endl;
    } else {
        std::cout << "Using Laplacian" << std::endl;
    }

    f.u.conservativeResize(f.u.rows(), potential_dirs.rows() + 1);
    f.u_grad.conservativeResize(f.u_grad.rows(), (potential_dirs.rows() * 3) + 3);
    f.v_theta.conservativeResize(f.v_theta.rows(), potential_dirs.rows() + 1);
    f.v_theta_grad.conservativeResize(f.v_theta_grad.rows(), (potential_dirs.rows() * 3) + 3);

    // pre-solve for Faraday effect, point charges

    std::unordered_map<int, int> global_to_matrix_ordering_pt_charge;
    Eigen::SparseMatrix<double> KKT_pt_charge;
    std::tie(global_to_matrix_ordering_pt_charge, KKT_pt_charge) = computeFaraday(f, pt_constraints, use_bilaplacian);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver_pt_charge(KKT_pt_charge);
    std::cout << "Initialized solver" << std::endl;

    // pre-solve for point charges (no shielding)
    // you don't actually need this and can remove it later

    std::unordered_map<int, int> global_to_matrix_ordering_base;
    Eigen::SparseMatrix<double> KKT_base;
    std::tie(global_to_matrix_ordering_base, KKT_base) = computeBasePotential(f, pt_constraints, use_bilaplacian);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::SquareSolver<double> solver_base(KKT_base);
    std::cout << "Initialized solver" << std::endl;

    // solve

    int i = potential_dirs.rows();

    double pt_const_val = f.v_theta.col(0).maxCoeff();
    Eigen::VectorXd pt_base = solveBasePotential(f, solver_base, global_to_matrix_ordering_base, pt_constraints, pt_const_val);
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

    Eigen::MatrixXd gradmag = Eigen::MatrixXd::Zero(f.TV.rows(), f.u.cols());
    f.max = Eigen::VectorXd::Zero(f.TV.rows());
    f.max_grad = Eigen::MatrixXd::Zero(f.TV.rows(), 3);

    for (int i = 0; i < f.u.cols(); i++) {
        std::cout << i << std::endl;
        Eigen::VectorXd grad_norm = (f.u_grad.middleCols(i * 3, 3)).rowwise().norm();
        // Eigen::VectorXd grad_diff_norm = (f.u_grad.middleCols(i * 3, 3) - f.v_theta_grad.middleCols(i * 3, 3)).rowwise().norm();
        gradmag.col(i) = grad_norm;
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

Eigen::SparseMatrix<double> computeFaraday(struct Faraday &f, bool use_bilaplacian) {

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

    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> D;
    Eigen::SparseMatrix<double> D_inv;

    igl::cotmatrix(f.TV, f.TT, M);
    igl::massmatrix(f.TV, f.TT, igl::MASSMATRIX_TYPE_BARYCENTRIC, D);
    igl::invert_diag(D, D_inv);

    Eigen::SparseMatrix<double> L = D_inv * M;
    if (use_bilaplacian) L = L * L;

    // split Laplacian into parts
    Eigen::SparseMatrix<double> L_NC(interior_count, interior_count);
    Eigen::SparseMatrix<double> L_CC(cage_count, cage_count);
    Eigen::SparseMatrix<double> L_NCC(interior_count, cage_count);
    Eigen::SparseMatrix<double> L_CNC(cage_count, interior_count);
    // std::vector<Eigen::Triplet<double>> L_NC_triplets;
    std::vector<Eigen::Triplet<double>> L_CC_triplets;
    std::vector<Eigen::Triplet<double>> L_NCC_triplets;
    std::vector<Eigen::Triplet<double>> L_CNC_triplets;

    int offset = interior_count + boundary_count;
    for (int k=0; k<L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
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
    
    std::cout << "Value is " << val_br << std::endl;

    for (int i = 0; i < interior_count; i++) {
        triplets.push_back(Eigen::Triplet<double>(interior_count, i, row_b(i)));
        triplets.push_back(Eigen::Triplet<double>(i, interior_count, col_r(i)));
    }
    triplets.push_back(Eigen::Triplet<double>(interior_count, interior_count, val_br));

    std::cout << "Setting triplets" << std::endl;

    LHS.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tDone" << std::endl;

    f.global_to_matrix_ordering = global_to_matrix_ordering;
    f.L = L;
    return LHS;

}

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, std::vector<int> &pt_constraints, double const_val) {

    // version for point constraints

    int constraints_size = f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt_constraints.size();

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(f.TV.rows() + constraints_size);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            int current_idx = global_to_matrix_ordering[i];
            RHS[f.TV.rows() + current_idx] = 0;
        }
    }
    for (size_t i = 0; i < pt_constraints.size(); i++) {
        RHS[f.TV.rows() + f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + i] = const_val;
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

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f, std::vector<int> &pt_constraints, bool use_bilaplacian) {

    // version for point constraints

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
    if (use_bilaplacian) L = L * L;

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
    for (int cage = 0; cage < f.is_cage_tv.sum() - 1; cage++) {
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + f.is_bdry_tv.sum() + cage, cage, 1.));
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + f.is_bdry_tv.sum() + cage, cage + 1, -1.));
        triplets.push_back(Eigen::Triplet<double>(cage, f.TV.rows() + f.is_bdry_tv.sum()+ cage, 1.));
        triplets.push_back(Eigen::Triplet<double>(cage + 1, f.TV.rows() + f.is_bdry_tv.sum() + cage, -1.));
    }
    for (int pt = 0; pt < pt_constraints.size(); pt++) {
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt, global_to_matrix_ordering[pt_constraints[pt]], 1.));
        triplets.push_back(Eigen::Triplet<double>(global_to_matrix_ordering[pt_constraints[pt]], f.TV.rows() + f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1 + pt, 1.));
    }


    std::cout << "\tAdded constraint triplets. About to set KKT matrix" << std::endl;

    KKT.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tSet KKT matrix. Size is " << KKT.rows() << " x " << KKT.cols() << ". Decomposing, this may a take a while..." << std::endl;

    return std::make_tuple(global_to_matrix_ordering, KKT);

}

Eigen::VectorXd solveBasePotential(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, std::vector<int> &pt_constraints, double const_val) {

    // meant for dealing with point charges

    int constraints_size = f.is_bdry_tv.sum() + pt_constraints.size();

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(f.TV.rows() + constraints_size);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            int current_idx = global_to_matrix_ordering[i];
            RHS[f.TV.rows() + current_idx] = 0;
        }
    }
    for (size_t i = 0; i < pt_constraints.size(); i++) {
        RHS[f.TV.rows() + f.is_bdry_tv.sum() + i] = const_val;
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

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeBasePotential(struct Faraday &f, std::vector<int> &pt_constraints, bool use_bilaplacian) {

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

    int constraints_size = f.is_bdry_tv.sum()+ pt_constraints.size();
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
    if (use_bilaplacian) L = L * L;

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
        triplets.push_back(Eigen::Triplet<double>(f.TV.rows() + f.is_bdry_tv.sum() + pt, global_to_matrix_ordering[pt_constraints[pt]], 1.));
        triplets.push_back(Eigen::Triplet<double>(global_to_matrix_ordering[pt_constraints[pt]], f.TV.rows() + f.is_bdry_tv.sum() + pt, 1.));
    }

    std::cout << "\tAdded constraint triplets. About to set KKT matrix" << std::endl;

    KKT.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tSet KKT matrix. Size is " << KKT.rows() << " x " << KKT.cols() << ". Decomposing, this may a take a while..." << std::endl;

    return std::make_tuple(global_to_matrix_ordering, KKT);

}