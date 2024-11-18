#include "solve.h"


Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::SparseMatrix<double> &grad, Eigen::VectorXd &func) {
    Eigen::VectorXd res = grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    return g_f;
}

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::SparseMatrix<double> &grad, Eigen::VectorXd &func) {
    Eigen::VectorXd res = grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    Eigen::MatrixXd g_tv = Eigen::MatrixXd::Zero(f.TV.rows(), 3);
    for (size_t i = 0; i < f.TV.rows(); i++) {
        for (int tet: f.my_tets[i]) {
            g_tv.row(i) += g_f.row(tet); // can weight by volume later
        }
        if (f.my_tets[i].size() > 0) g_tv.row(i) /= f.my_tets[i].size();
    }
    return g_tv;
}

Eigen::VectorXd solveFaraday(struct Faraday &f, Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, Eigen::VectorXd &bdry_vals) {

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
    if (solver.info() != Eigen::Success) {
        std::cout << "ERROR: Solve failed!" << std::endl;
        exit(-1);
    }

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());

    for (size_t i = 0; i < f.TV.rows(); i++) sol[i] = u[global_to_matrix_ordering[i]];

    return sol;
}

std::unordered_map<int, int> computeFaraday(struct Faraday &f, Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver) {

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

    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(f.TV, f.TT, L);
    L = L * L;

    // put mass matrix here

    for (int k=0; k<L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
            triplets.push_back(Eigen::Triplet<double>(global_to_matrix_ordering[it.row()], global_to_matrix_ordering[(it.col())], it.value()));
            // std::cout << global_to_matrix_ordering[it.row()] << " " << global_to_matrix_ordering[(it.col())] << " " << it.value() << "\n";
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

    solver.compute(KKT);
    if (solver.info() != Eigen::Success) {
        std::cout << "ERROR: Decomposition failed!" << std::endl;
        exit(-1);
    }
    
    return global_to_matrix_ordering;

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