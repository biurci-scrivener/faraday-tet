#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"
#include "args/args.hxx"
#include "io.h"
#include "pc.h"
#include "vis.h"
#include "faraday.h"
#include "solve.h"

#include <igl/grad.h>
#include <igl/barycenter.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/volume.h>

#include "imgui.h"

struct Faraday f;
int u_idx = 0;
int pt_idx = 0;
std::vector<int> pt_constraints;

void myCallback() {
	ImGui::PushItemWidth(100);

	ImGui::Text("View direction: ");
	ImGui::SameLine();
	if (ImGui::InputInt("##Dir", &u_idx)) {
		if (u_idx < 0) {
			u_idx = 0;
		} else if (u_idx >= f.u.cols()) {
			u_idx = f.u.cols() - 1;
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Update")) {
		vis_u(f, u_idx);
  	}

	ImGui::Text("");

	ImGui::Text("Add point charge at: ");
	ImGui::SameLine();
	ImGui::InputInt("##PtCharge", &pt_idx);
	if (ImGui::Button("Add")) {
		if ((pt_idx >= 0) && (pt_idx < f.TV.rows()) && (!f.is_bdry_tv(pt_idx)) && (!f.is_cage_tv(pt_idx))) {
			auto it = std::find(pt_constraints.begin(), pt_constraints.end(), pt_idx);
			if (it == pt_constraints.end()) {
				pt_constraints.push_back(pt_idx);
			}
		} else {
			std::cout << "Bad index for pt. constraint" << std::endl;
		}
	}
	std::string list_pt_charges = "Currently, point charges at: \n";
	for (int pt: pt_constraints) list_pt_charges += std::to_string(pt) + "\n";
	ImGui::Text("%s", list_pt_charges.c_str());
	if (ImGui::Button("Recompute")) {

		solvePotentialPointCharges(f, pt_constraints);
		solveFieldDifference(f);
		estimateNormals(f);

		Eigen::VectorXd flipped = scoreNormalEst(f);

		polyscope::PointCloud * pc = polyscope::getPointCloud("Points");
		pc->addVectorQuantity("Normals, true", f.N);
		pc->addVectorQuantity("Normals, est.", f.N_est);
		pc->addScalarQuantity("Flipped", flipped);

		polyscope::PointCloud * tet_pc = polyscope::getPointCloud("Tet. mesh, verts.");
		Eigen::VectorXi is_point_constrained = Eigen::VectorXi::Zero(f.TV.rows());
		for (int pt_constrained: pt_constraints) is_point_constrained[pt_constrained] = 1;
		tet_pc->addScalarQuantity("Constrained pts.", is_point_constrained);

		vis_max(f);
  	}
	if (ImGui::Button("Clear selected")) {
		pt_constraints.clear();
	}


	ImGui::PopItemWidth();
}

int main(int argc, char **argv) {

	// Configure the argument parser
	args::ArgumentParser parser("3D Faraday cage test project");
	args::Positional<std::string> inputFilename(parser, "pc", "A point cloud");

	// Parse args
	try {
	parser.ParseCLI(argc, argv);
	} catch (args::Help &h) {
	std::cout << parser;
	return 0;
	} catch (args::ParseError &e) {
	std::cerr << e.what() << std::endl;
	std::cerr << parser;
	return 1;
	}

	if (!inputFilename) {
	std::cerr << "Please specify a mesh file as argument" << std::endl;
	return EXIT_FAILURE;
	}

	std::string filename = args::get(inputFilename);

	Eigen::MatrixXd P_original;
	Eigen::MatrixXd N_original;

	// load point cloud 

	std::tie(P_original, N_original) = parsePLY(filename);

	std::cout << "Loaded file " << filename << std::endl;
	
	f.P = P_original;
	f.N = N_original;

	prepareTetgen(f);

	std::cout << "Starting tetrahedralization..." << std::endl;

	double tet_area = 0.0001 * (f.bb.row(0) - f.bb.row(1)).cwiseAbs().prod();

	std::cout << "Max tet area: " << tet_area << std::endl;
	std::cout << (f.bb.row(0) - f.bb.row(1)).cwiseAbs() << std::endl;
	std::cout << (f.bb.row(0) - f.bb.row(1)).cwiseAbs().prod() << std::endl;

	if (igl::copyleft::tetgen::tetrahedralize(	f.V, f.F, f.H, f.VM, f.FM, f.R, "pq1.414a"+ std::to_string(tet_area),
												f.TV, f.TT, f.TF, f.TM, f.TR, f.TN, f.PT, f.FT, f.num_regions)) exit(-1);

	// if (igl::copyleft::tetgen::tetrahedralize(	f.V, f.F, f.H, f.VM, f.FM, f.R, "p",
	// 											f.TV, f.TT, f.TF, f.TM, f.TR, f.TN, f.PT, f.FT, f.num_regions)) exit(-1);

	std::cout << "Finished tetrahedralizing" << std::endl;
	std::cout << "Computing cell barycenters" << std::endl;
	igl::barycenter(f.TV, f.TT, f.BC);
	std::cout << "Computing grad operator" << std::endl;
	igl::grad(f.TV, f.TT, f.grad);
	std::cout << "Computing cell volumes" << std::endl;
	igl::volume(f.TV, f.TT, f.vols);
	std::cout << "Assigning boundary/cage verts." << std::endl;
	findBdryCage(f);
	std::cout << "Assigning tets." << std::endl;
	findTets(f);

	// solve for field over many directions

	solvePotentialOverDirs(f);
	solveFieldDifference(f);
	estimateNormals(f);

	Eigen::VectorXd flipped = scoreNormalEst(f);

	polyscope::init();
	polyscope::state::userCallback = myCallback;

	// initialize structures

	auto pc = polyscope::registerPointCloud("Points", f.P);
	pc->setEnabled(true);
	pc->addVectorQuantity("Normals, true", f.N);
	pc->addVectorQuantity("Normals, est.", f.N_est);
	pc->addScalarQuantity("Flipped", flipped);

	auto tet_mesh = polyscope::registerTetMesh("Tet. mesh", f.TV, f.TT);
	tet_mesh->setCullWholeElements(false);
	tet_mesh->setEnabled(true);

	// Gurobi test
	// Eigen::VectorXd sol_gurobi = solvePotentialOverDirs_Gurobi(f);
	// tet_mesh->addVertexScalarQuantity("Gurobi solve", sol_gurobi);

	auto tet_slice = polyscope::addSceneSlicePlane();
	tet_slice->setDrawPlane(false);
	tet_slice->setDrawWidget(true);
	tet_slice->setVolumeMeshToInspect("Tet. mesh");

	auto tv_vis = polyscope::registerPointCloud("Tet. mesh, verts.", f.TV);
	auto bc_vis = polyscope::registerPointCloud("Tet. mesh, cell centers", f.BC);

	// call vis. functions

	tv_vis->addScalarQuantity("Bdry.", f.is_bdry_tv);
	tv_vis->addScalarQuantity("Cage", f.is_cage_tv);

	vis_max(f);

	polyscope::show();

	return EXIT_SUCCESS;

}