#include "solve.h"

Eigen::MatrixXd potential_dirs = ico_pts_2;

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

void solvePotentialOverDirs(struct Faraday &f) {
    

    std::cout << "Starting solve for potential over all directions." << std::endl;

    f.u = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows());
    f.u_grad = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows() * 3);
    f.v_theta = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows());
    f.v_theta_grad = Eigen::MatrixXd::Zero(f.TV.rows(), potential_dirs.rows() * 3);

    std::unordered_map<int, int> global_to_matrix_ordering;
    Eigen::SparseMatrix<double> KKT;
    std::tie(global_to_matrix_ordering, KKT) = computeFaraday(f);

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
        std::cout << "\tSolving for directions " << i << std::endl;

        Eigen::VectorXd boundary_vals(f.TV.rows());
        Eigen::VectorXd dir = potential_dirs.row(i);
        for (int j = 0; j < f.TV.rows(); j++) boundary_vals[j] = f.TV.row(j).dot(dir);
        f.v_theta.col(i) = boundary_vals;
        Eigen::MatrixXd boundary_vals_grad = grad_tv(f, boundary_vals);
        f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
        f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
        f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

        Eigen::VectorXd res = solveFaraday(f, solver, global_to_matrix_ordering, boundary_vals);
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

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::SquareSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, Eigen::VectorXd &bdry_vals) {

    int constraints_size = f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1;

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(f.TV.rows() + constraints_size);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            int current_idx = global_to_matrix_ordering[i];
            RHS[f.TV.rows() + current_idx] = bdry_vals[i];
        }
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

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f) {

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

    int constraints_size = f.is_bdry_tv.sum() + f.is_cage_tv.sum() - 1;
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

    std::cout << "\tAdded constraint triplets. About to set KKT matrix" << std::endl;

    KKT.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tSet KKT matrix. Size is " << KKT.rows() << " x " << KKT.cols() << ". Decomposing, this may a take a while..." << std::endl;

    return std::make_tuple(global_to_matrix_ordering, KKT);

}

// Eigen::VectorXd solveDirichletProblem(struct Faraday &f, Eigen::VectorXd &bdry_vals) {

//     Eigen::VectorXd sol = Eigen::VectorXd::Zero((f.TV.rows()));

//     size_t BDRY_END = f.is_boundary_point.sum();

//     Eigen::SparseMatrix<double> L, L_in_in, L_in_b;

//     igl::cotmatrix(f.TV, f.TT, L);

//     Eigen::VectorXi bdry_indices = Eigen::VectorXi::LinSpaced(BDRY_END, 0, BDRY_END - 1);
//     std::cout << bdry_indices.transpose() << std::endl;

//     Eigen::VectorXi interior_indices = Eigen::VectorXi::LinSpaced(f.TV.rows() - BDRY_END, BDRY_END, f.TV.rows() - 1);
//     std::cout << interior_indices.transpose() << std::endl;
//     Eigen::VectorXd g;

//     // std::cout << L << std::endl;

//     igl::slice(L, interior_indices, interior_indices, L_in_in);
//     igl::slice(L, interior_indices, bdry_indices, L_in_b);
//     igl::slice(bdry_vals, bdry_indices, g);

//     Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

//     std::cout << "Starting Dirichlet solve..." << std::endl;
//     solver.compute(-L_in_in);
//     sol(interior_indices) = solver.solve(L_in_b * g);
//     sol(bdry_indices) = g;
//     std::cout << "Finished Dirichlet solve" << std::endl;

//     return sol;
// }