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

std::tuple<Eigen::VectorXi, Eigen::VectorXi> findBdryCage(struct Faraday &f) {

    Eigen::VectorXi is_bdry_tv = Eigen::VectorXi::Zero(f.TV.rows());
    Eigen::VectorXi is_cage_tv = Eigen::VectorXi::Zero(f.TV.rows());

    for (size_t i = 0; i < f.TV.rows(); i++) {
        if ((i < f.is_cage_point.rows()) && (i > 7)) {
            is_cage_tv[i] = f.is_cage_point(i);
        } else if ( (f.TV(i,0) == f.bb(0,0) || f.TV(i,0) == f.bb(1,0)) ||
                    (f.TV(i,1) == f.bb(0,1) || f.TV(i,1) == f.bb(1,1)) ||
                    (f.TV(i,2) == f.bb(0,2) || f.TV(i,2) == f.bb(1,2))
                    ) {
            is_bdry_tv[i] = true;
        }
    }

    return std::make_tuple(is_bdry_tv, is_cage_tv);

}

std::vector<std::vector<int>> findTets(struct Faraday &f) {

    std::vector<std::vector<int>> my_tets(f.TV.rows(), std::vector<int>());

    for (size_t i = 0; i < f.TT.rows(); i++) {
        for (int vtx: f.TT.row(i)) {
            my_tets[vtx].push_back(i);
        }
    }

    return my_tets;

}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXi, std::vector<std::vector<int>>, Eigen::MatrixXd, Eigen::MatrixXi> appendBoundaryAndCage(Eigen::MatrixXd &P, Eigen::MatrixXd &N) {

    // pre-append corners of bounding box

    Eigen::MatrixXd BV;
    Eigen::MatrixXi BF;

    igl::bounding_box(P, BV, BF);

    double PADDING = 0.25;

    Eigen::Vector3d bb_max = BV.row(0);
    Eigen::Vector3d bb_min = BV.row(7);

    double pad = (bb_max - bb_min).cwiseAbs().minCoeff() * PADDING;

    igl::bounding_box(P, pad, BV, BF);
    
    Eigen::MatrixXd bb(2,3);
    bb.row(0) << BV.row(0);
    bb.row(1) << BV.row(7);

    int START_BDRY = 0;

    std::vector<Eigen::Vector3d> add_rows;

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

        // refine each square face
        // size_t REFINE_DEG = 1;

        // // bottom 3
        // Eigen::Vector3d base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     for (size_t j = 1; j < REFINE_DEG; j++) {
        //         add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1] + (j * (delta[1] / REFINE_DEG)), base[2]});
        //     }
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     for (size_t j = 1; j < REFINE_DEG; j++) {
        //         add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2] + (j * (delta[2] / REFINE_DEG))});
        //     }
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     for (size_t j = 1; j < REFINE_DEG; j++) {
        //         add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2] + (j * (delta[2] / REFINE_DEG))});
        //     }
        // }
        // // top 3
        // base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     for (size_t j = 1; j < REFINE_DEG; j++) {
        //         add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1] + (j * (delta[1] / REFINE_DEG)), base[2]});
        //     }
        // }
        // base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     for (size_t j = 1; j < REFINE_DEG; j++) {
        //         add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2] + (j * (delta[2] / REFINE_DEG))});
        //     }
        // }
        // base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     for (size_t j = 1; j < REFINE_DEG; j++) {
        //         add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2] + (j * (delta[2] / REFINE_DEG))});
        //     }
        // }
        
        // // refine 12 edges
        // base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        // }
        // base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        // }
        // base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        // }
        // base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        // }
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        // }
        // base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_max[2] + pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        // }
        // base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        // }
        // base = { bb_max[0] + pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        // for (size_t i = 1; i < REFINE_DEG; i++) {
        //     add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        // }
    
    int START_CAGE = add_rows.size();

    // ADD CAGE POINTS
    // Icosphere surrounding each interior point

    double CAGE_RADIUS = pad / 5.;

    for (int i = 0; i < P.rows(); i++) {

        Eigen::Vector3d pt = P.row(i);

        for (int j = 0; j < ico_pts.rows(); j++) {
            Eigen::Vector3d ico_pt = ico_pts.row(j);
            add_rows.push_back(pt + ico_pt * CAGE_RADIUS);
        }

    }

    // append points
    Eigen::MatrixXd P_new(P.rows() + add_rows.size(), 3);
    Eigen::MatrixXd N_new = Eigen::MatrixXd::Zero(N.rows() + add_rows.size(), 3);
    Eigen::VectorXi is_cage_point = Eigen::VectorXi::Zero(P.rows() + add_rows.size());
    std::vector<std::vector<int>> my_cage_points(P.rows() + add_rows.size(), std::vector<int>());
    
    size_t i = 0;
    for (Eigen::Vector3d row_to_add: add_rows) {
        P_new.row(i) = row_to_add;
        if (i >= START_CAGE) {
            is_cage_point[i] = 1;
            my_cage_points[((i - START_CAGE) / 12) + add_rows.size()].push_back(i);
        }
        i++;
    }
    while (i < P.rows() + add_rows.size()) {
        P_new.row(i) = P.row(i - add_rows.size());
        N_new.row(i) = N.row(i - add_rows.size());
        i++;
    }

    return std::make_tuple(P_new, N_new, is_cage_point, my_cage_points, bb, BF);

}

