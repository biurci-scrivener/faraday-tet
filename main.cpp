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

struct Faraday f;
int u_idx = 0;

void myCallback() {
	ImGui::PushItemWidth(100);
	if (ImGui::InputInt("Currently viewing: ", &u_idx)) {
		if (u_idx < 0) {
			u_idx = 0;
		} else if (u_idx >= f.u.cols()) {
			u_idx = f.u.cols() - 1;
		}
	}
	if (ImGui::Button("Update")) {
		vis_u(f, u_idx);
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

	double tet_area = 0.00001 * pow((f.bb.row(0) - f.bb.row(1)).cwiseAbs().prod(), 1./3.);

	std::cout << "Max tet area: " << tet_area << std::endl;

	if (igl::copyleft::tetgen::tetrahedralize(	f.V, f.F, f.H, f.VM, f.FM, f.R, "Vpq1.414a"+ std::to_string(tet_area),
												f.TV, f.TT, f.TF, f.TM, f.TR, f.TN, f.PT, f.FT, f.num_regions)) exit(-1);

	std::cout << "Finished tetrahedralizing" << std::endl;
	igl::barycenter(f.TV, f.TT, f.BC);
	igl::grad(f.TV, f.TT, f.grad);
	igl::volume(f.TV, f.TT, f.vols);
	findBdryCage(f);
	findTets(f);

	// solve for field over many directions

	solvePotentialOverDirs(f);
	solveFieldDifference(f);
	estimateNormals(f);

	polyscope::init();
	polyscope::state::userCallback = myCallback;

	// initialize structures

	auto pc = polyscope::registerPointCloud("Points", f.P);
	pc->setEnabled(true);
	pc->addVectorQuantity("Normals, true", f.N);
	pc->addVectorQuantity("Normals, est.", f.N_est);

	auto tet_mesh = polyscope::registerTetMesh("Tet. mesh", f.TV, f.TT);
	tet_mesh->setCullWholeElements(false);
	tet_mesh->setEnabled(true);
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