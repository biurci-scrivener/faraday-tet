#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "args/args.hxx"
#include "io.h"
#include "pc.h"
#include "faraday.h"
#include "solve.h"

#include <igl/grad.h>
#include <igl/barycenter.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> faraday_solver;

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

	// Make sure a mesh name was given
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

	struct Faraday faraday;

	Eigen::MatrixXi bb_faces;
	std::tie(faraday.P, faraday.N, faraday.is_cage_point, faraday.my_cage_points, faraday.bb, bb_faces) = appendBoundaryAndCage(P_original, N_original);
	std::cout << "Starting tetrahedralization..." << std::endl;
	if (igl::copyleft::tetgen::tetrahedralize(faraday.P, bb_faces, "pq1.414a0.01", faraday.TV, faraday.TT, faraday.TF)) {
		exit(-1);
	}
	
	std::cout << "Finished tetrahedralizing" << std::endl;
	igl::barycenter(faraday.TV, faraday.TT, faraday.BC);
	std::tie(faraday.is_bdry_tv, faraday.is_cage_tv) = findBdryCage(faraday);
	faraday.my_tets = findTets(faraday);

	Eigen::SparseMatrix<double> grad;
	igl::grad(faraday.TV, faraday.TT, grad);

	polyscope::init();

	auto pc = polyscope::registerPointCloud("Points", faraday.P);
	pc->setEnabled(true);

	auto tet_mesh = polyscope::registerTetMesh("Tet. mesh", faraday.TV, faraday.TT);
	tet_mesh->setCullWholeElements(false);
	tet_mesh->setEnabled(true);
	auto tet_slice = polyscope::addSceneSlicePlane();
	tet_slice->setDrawPlane(false);
	tet_slice->setDrawWidget(true);
	tet_slice->setVolumeMeshToInspect("Tet. mesh");

	polyscope::show();

	return EXIT_SUCCESS;

}