#include "vis.h"

void vis_u(struct Faraday f, int idx) {
    polyscope::VolumeMesh * tet_mesh = polyscope::getVolumeMesh("Tet. mesh");
    auto v_theta = tet_mesh->addVertexScalarQuantity("v_theta", f.v_theta.col(idx));
    v_theta->setEnabled(true);
    auto u = tet_mesh->addVertexScalarQuantity("u", f.u.col(idx));
    u->setEnabled(true);

    polyscope::PointCloud * tet_pc = polyscope::getPointCloud("Tet. mesh, verts.");
    tet_pc->addVectorQuantity("u, grad.", f.u_grad.middleCols(idx * 3, 3));
    tet_pc->addVectorQuantity("v_theta, grad.", f.v_theta_grad.middleCols(idx * 3, 3));
}

void vis_max(struct Faraday f) {

    polyscope::VolumeMesh * tet_mesh = polyscope::getVolumeMesh("Tet. mesh");
    auto u = tet_mesh->addVertexScalarQuantity("max_{theta} ||grad{v_theta} - grad{u_theta}||",  f.max);
    u->setEnabled(true);


    polyscope::PointCloud * tet_pc = polyscope::getPointCloud("Tet. mesh, verts.");
    tet_pc->addScalarQuantity("max_{theta} ||grad{v_theta} - grad{u_theta}||", f.max);
    tet_pc->addVectorQuantity("Grad. of max.", f.max_grad);
    tet_pc->addVectorQuantity("Grad. of max., normalized", f.max_grad_normalized);

    // put them just on the cage vertices, also
    Eigen::MatrixXd max_grad_cage = f.max_grad; // should be a deep copy
    Eigen::MatrixXd max_grad_normalized_cage = f.max_grad_normalized; 
    for (size_t i; i < f.TV.rows(); i++) {
        if (!f.is_cage_tv(i)) {
            max_grad_cage.row(i) = Eigen::VectorXd::Zero(3);
            max_grad_normalized_cage.row(i) = Eigen::VectorXd::Zero(3);
        }
    }
    tet_pc->addVectorQuantity("Grad. of max. (cage)", max_grad_cage);
    tet_pc->addVectorQuantity("Grad. of max., normalized (cage)", max_grad_normalized_cage);


}