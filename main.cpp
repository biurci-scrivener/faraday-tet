#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "args/args.hxx"
#include "io.h"
#include "pc.h"

#include <igl/grad.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

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
	std::cout << filename << std::endl;

	Eigen::MatrixXd P_original;
	Eigen::MatrixXd N_original;

	// load point cloud 
	std::tie(P_original, N_original) = parsePLY(filename);

	std::cout << "Loaded file" << std::endl;

	Eigen::MatrixXd P;
	Eigen::MatrixXd N;
	Eigen::VectorXi is_boundary_point;
	Eigen::VectorXi is_cage_point;
	std::vector<std::vector<int>> my_cage_points;
	Eigen::MatrixXd bb;

	std::tie(P, N, is_boundary_point, is_cage_point, my_cage_points, bb) = appendBoundaryAndCage(P_original, N_original);

	Eigen::MatrixXd TV;
	Eigen::MatrixXi TT;
	Eigen::MatrixXi TF;
	Eigen::MatrixXi F;
	igl::copyleft::tetgen::tetrahedralize(P, F, "pq1.414c", TV, TT, TF);

	Eigen::SparseMatrix<double> grad;
	igl::grad(TV, TT, grad);

	polyscope::init();

	Eigen::VectorXd base_field = Eigen::VectorXd::Zero(TV.rows());
	Eigen::Vector3d dir = {1,1,1};
	for (size_t i = 0; i < TV.rows(); i++) {
		base_field[i] = TV.row(i).dot(dir);
	}

	Eigen::VectorXd res = grad * base_field;
	Eigen::MatrixXd base_field_grad = Eigen::Map<Eigen::MatrixXd>(res.data(), TT.rows(), 3);

	auto pc = polyscope::registerPointCloud("Points", P);
	pc->addVectorQuantity("True Normals", N);
	pc->setEnabled(true);

	auto tet_mesh = polyscope::registerTetMesh("Tet. mesh", TV, TT);
	auto base_field_q = tet_mesh->addVertexScalarQuantity("Base field", base_field);
	auto base_field_grad_q = tet_mesh->addCellVectorQuantity("Base field, grad", base_field_grad);
	base_field_q->setEnabled(true);
	tet_mesh->setEnabled(true);
	tet_mesh->setCullWholeElements(false);

	size_t debug_vertex = 0;
	while (is_boundary_point(debug_vertex) || is_cage_point(debug_vertex)) debug_vertex++;
	Eigen::VectorXd cage_debug = Eigen::VectorXd::Zero(P.rows());
	for (int cage: my_cage_points[debug_vertex]) cage_debug[cage] = 1;
	pc->addScalarQuantity("Cage vertices (debug) " + std::to_string(debug_vertex), cage_debug);
	pc->addScalarQuantity("Boundary vertices", is_boundary_point);
	pc->addScalarQuantity("Cage vertices", is_cage_point);

	polyscope::show();

	return EXIT_SUCCESS;

}