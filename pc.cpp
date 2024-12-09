#include "pc.h"

bool is_close(double a, double b) {
    return fabs(a - b) < 1e-12;
}

template <typename T> std::vector<int> sort_indexes(const std::vector<T> &v) {

    // from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes

    // initialize original index locations
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    std::stable_sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) {return v[i1] < v[i2];});

    return idx;
}

template <typename T> std::vector<T> reorder_vector(const std::vector<T> &vals, const std::vector<int> &idxs) {
    std::vector<T> vals_new;
    for (int idx: idxs) {vals_new.push_back(vals[idx]);}
    return vals_new;
}

Eigen::VectorXd scoreNormalEst(struct Faraday &f) {

    double sum_dot = 0;
    size_t sum_disagree = 0;
    Eigen::VectorXd flipped = Eigen::VectorXd::Zero(f.N.rows());
    for (size_t i = 0; i < f.N.rows(); i++) {
        double dp = f.N.row(i).dot(f.N_est.row(i));
        sum_dot += dp;
        sum_disagree += dp < 0;
        flipped[i] = dp < 0;
    }

    std::cout << std::endl;
    std::cout << "Total number flipped: " << sum_disagree << std::endl;
    std::cout << "Percent agree: " << ((double)(f.N.rows() - sum_disagree)) / f.N.rows() << std::endl;
    std::cout << "Avg. dot product: " << sum_dot / f.N.rows() << std::endl;
    std::cout << std::endl;

    return flipped;

}

void findBdryCage(struct Faraday &f) {

    Eigen::VectorXi is_bdry_tv_new = Eigen::VectorXi::Zero(f.TV.rows());
    Eigen::VectorXi is_cage_tv_new = Eigen::VectorXi::Zero(f.TV.rows());

    for (size_t i = 0; i < f.TV.rows(); i++) {
        if ((i < f.is_cage_tv.rows()) && (i > 7)) {
            is_cage_tv_new[i] = f.is_cage_tv(i);
        } else if ( (f.TV(i,0) == f.bb(0,0) || f.TV(i,0) == f.bb(1,0)) ||
                    (f.TV(i,1) == f.bb(0,1) || f.TV(i,1) == f.bb(1,1)) ||
                    (f.TV(i,2) == f.bb(0,2) || f.TV(i,2) == f.bb(1,2))
                    ) {
            is_bdry_tv_new[i] = true;
        }
    }

    f.is_cage_tv = is_cage_tv_new;
    f.is_bdry_tv = is_bdry_tv_new;

}

void findTets(struct Faraday &f) {

    std::vector<std::vector<int>> my_tets(f.TV.rows(), std::vector<int>());

    for (size_t i = 0; i < f.TT.rows(); i++) {
        for (int vtx: f.TT.row(i)) {
            my_tets[vtx].push_back(i);
        }
    }

    f.my_tets = my_tets;

}

void prepareTetgen(struct Faraday &f, double cr_factor) {

    // pre-append corners of bounding box

    Eigen::MatrixXd BV;
    Eigen::MatrixXi BF;

    igl::bounding_box(f.P, BV, BF);

    double PADDING = 0.25;

    Eigen::Vector3d bb_max = BV.row(0);
    Eigen::Vector3d bb_min = BV.row(7);

    double pad = (bb_max - bb_min).cwiseAbs().minCoeff() * PADDING;

    igl::bounding_box(f.P, pad, BV, BF);
    
    Eigen::MatrixXd bb(2,3);
    bb.row(0) << BV.row(0);
    bb.row(1) << BV.row(7);

    int START_BDRY = 0;

    std::vector<Eigen::Vector3d> add_rows;
    std::vector<Eigen::Vector3i> add_faces;

    // ADD BOUNDARY POINTS

    // corners
    add_rows.push_back(BV.row(0));
    add_rows.push_back(BV.row(1));
    add_rows.push_back(BV.row(2));
    add_rows.push_back(BV.row(3));
    add_rows.push_back(BV.row(4));
    add_rows.push_back(BV.row(5));
    add_rows.push_back(BV.row(6));
    add_rows.push_back(BV.row(7));
    add_faces.push_back(BF.row(0));
    add_faces.push_back(BF.row(1));
    add_faces.push_back(BF.row(2));
    add_faces.push_back(BF.row(3));
    add_faces.push_back(BF.row(4));
    add_faces.push_back(BF.row(5));
    add_faces.push_back(BF.row(6));
    add_faces.push_back(BF.row(7));
    add_faces.push_back(BF.row(8));
    add_faces.push_back(BF.row(9));
    add_faces.push_back(BF.row(10));
    add_faces.push_back(BF.row(11));
    
    int START_CAGE = add_rows.size();

    // ADD CAGE POINTS
    // Icosphere surrounding each interior point

    double CAGE_RADIUS = pad / cr_factor;
    std::cout << "Cage radius is " << CAGE_RADIUS << std::endl;
    std::cout << "\tFactor: " << cr_factor << std::endl;

    for (int i = 0; i < f.P.rows(); i++) {

        Eigen::Vector3d pt = f.P.row(i);

        size_t this_base = add_rows.size();

        for (int j = 0; j < ico_pts.rows(); j++) {
            Eigen::Vector3d ico_pt = ico_pts.row(j);
            add_rows.push_back(pt + ico_pt * CAGE_RADIUS);
        }

        for (int j = 0; j < ico_faces.rows(); j++) {
            Eigen::Vector3i ico_face = ico_faces.row(j);
            add_faces.push_back(Eigen::Vector3i::Ones() * this_base + ico_face);
        }

    }

    // append points
    Eigen::MatrixXd V(add_rows.size(), 3);
    Eigen::MatrixXi F(add_faces.size(), 3);
    Eigen::VectorXi is_cage_tv(add_rows.size());
    std::vector<std::vector<int>> my_cage_points(f.P.size(), std::vector<int>());
    
    size_t i = 0;
    for (Eigen::Vector3d row_to_add: add_rows) {
        V.row(i) = row_to_add;
        if (i >= START_CAGE) {
            is_cage_tv[i] = 1;
            my_cage_points[(i - START_CAGE) / 12].push_back(i);
        }
        i++;
    }

    i = 0;
    for (Eigen::VectorXi face_to_add: add_faces) {
        F.row(i) = face_to_add;
        i++;
    }

    // set attributes of Faraday struct
    f.bb = bb;
    f.my_cage_points = my_cage_points;
    f.is_cage_tv = is_cage_tv;
    f.V = V;
    f.F = F;
    f.H = f.P;

}

