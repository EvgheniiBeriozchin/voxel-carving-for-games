#include <iostream>
#include <opencv2/opencv.hpp>

#include "voxel/VoxelGrid.h"
#include "voxel/VoxelGridExporter.h"
#include "voxel/SpaceCarver.h"
#include "Camera.h"
#include "utils/utils.h"
#include "TutteEmbedding.h"

#include <string>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/writePLY.h>
#include <igl/edges.h>

#define RUN_CAMERA_CALIBRATION 0
#define RUN_POSE_ESTIMATION_TEST 0
#define RUN_VOXEL_GRID_TEST 0
#define RUN_VOXEL_CARVING 0
#define RUN_UV_TEST 1


const int NUM_PROCESSED_FRAMES = 1000;
const std::string CALIBRATION_VIDEO_NAME = "../Box_NaturalLight.mp4";
const std::string RECONSTRUCTION_VIDEO_NAME = "../PepperMill_NaturalLight.mp4";
const std::string voxeTestFilenameTarget = std::string("voxelGrid.off");
const std::string uvTestingInput = "../beetle.obj";

void writeObj(std::string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N)
{
	std::ofstream myfile;
	myfile.open(filename);
	myfile << "o " << filename << std::endl;
	for (int i = 0; i < V.rows(); i++)
	{
		myfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
	}
	for (int i = 0; i < U.rows(); i++)
	{
		myfile << "vt " << U(i, 0) << " " << U(i, 1) << std::endl;
	}
	for (int i = 0; i < F.rows(); i++)
	{
		myfile << "f " << F(i, 0) + 1 << "/" << F(i, 0) + 1 << " " << F(i, 1) + 1 << "/" << F(i, 1) + 1 << " " << F(i, 2) + 1 << "/" << F(i, 2) + 1 << std::endl;
	}

	if (N.rows() > 0)
	{
		for (int i = 0; i < N.rows(); i++)
		{
			myfile << "vn " << N(i, 0) << " " << N(i, 1) << " " << N(i, 2) << std::endl;
		}
	}
	myfile.close();
}

int main() {
	cv::Mat cameraMatrix, distanceCoefficients;
	cv::VideoCapture calibrationVideo(CALIBRATION_VIDEO_NAME), reconstructionVideo(RECONSTRUCTION_VIDEO_NAME);
	cv::aruco::ArucoDetector detector = createDetector();
	cv::aruco::Board* board = createBoard();

	if (RUN_CAMERA_CALIBRATION)
	{
		if (!videoExists(calibrationVideo))
		{
			return 0;
		}

		calibrateCamera(calibrationVideo, &detector, board, &cameraMatrix, &distanceCoefficients);
	}

	if (RUN_POSE_ESTIMATION_TEST)
	{
		if (!videoExists(reconstructionVideo))
		{
			return 0;
		}

		while (reconstructionVideo.grab())
		{
			cv::Mat image;
			reconstructionVideo.retrieve(image);

			Camera frame = Camera(image, cameraMatrix);
			Pose currentPose = frame.estimateCameraPose(&detector, board, cameraMatrix, distanceCoefficients);

			std::cout << "Camera translation vector: " << currentPose.position.x() << ", "
				<< currentPose.position.y() << ", " << currentPose.position.z() << std::endl;
			std::cout << "Camera rotation vector: " << currentPose.rotation.x() << ", "
				<< currentPose.rotation.y() << ", " << currentPose.rotation.z() << std::endl;
		}
	}

	if (RUN_VOXEL_GRID_TEST) {
		auto grid = VoxelGrid::CreateFilledVoxelGrid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3i(50, 50, 50), 1);
		std::cout << grid.GetVoxelCount() << std::endl;
		auto v = grid.GetVoxelCenter(Eigen::Vector3i(1, 0, 0));
		for (int i = 0; i < 50; i++) {
			for (int j = 30; j < 40; j++) {
				for (int k = 30; k < 40; k++) {
					grid.RemoveVoxel(Eigen::Vector3i(i, j, k));
				}
			}
		}
		VoxelGridExporter::ExportToOFF(voxeTestFilenameTarget, grid);
	}

	if (RUN_VOXEL_CARVING) 
	{
		if (!videoExists(reconstructionVideo))
		{
			return 0;
		}

		std::vector<Camera> cameraFrames;
		int numFrames = reconstructionVideo.get(cv::CAP_PROP_FRAME_COUNT);
		auto grid = VoxelGrid::CreateFilledVoxelGrid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3i(50, 50, 50), 1);

		for (int i = 0; i < NUM_PROCESSED_FRAMES; i++)
		{
			cv::Mat image;
			reconstructionVideo.set(1, i * (numFrames / NUM_PROCESSED_FRAMES));
			reconstructionVideo.retrieve(image);

			cameraFrames.push_back(Camera(image, cameraMatrix));
		}

		SpaceCarver::MultiSweep(grid, cameraFrames);
		VoxelGridExporter::ExportToOFF(voxeTestFilenameTarget, grid);
	}

	if (RUN_UV_TEST) {
		Eigen::MatrixXd V, U_tutte, U, N;
		Eigen::MatrixXi F;

		igl::read_triangle_mesh(uvTestingInput, V, F);

		TutteEmbedder::GenerateUvMapping(V, F, U, N);

		igl::opengl::glfw::Viewer viewer;

		viewer.data().set_mesh(V, F);
		viewer.data().set_colors(N.array() * 0.5 + 0.5);
		viewer.data().show_texture = true;
		viewer.data().show_lines = false;

		writeObj("../beetleOut.obj", V, F, U_tutte, N);


		viewer.launch();
	}
	
	return 0;
}
 
/*
void tutte(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& U)
{
	Eigen::VectorXi bL;
	igl::boundary_loop(F, bL);

	Eigen::MatrixXd UV;
	igl::map_vertices_to_circle(V, bL, UV);

	Eigen::SparseMatrix<double> L(V.rows(), V.rows());
	igl::cotmatrix(V, F, L);

	igl::min_quad_with_fixed_data<double> data;
	igl::min_quad_with_fixed_precompute(L, bL, Eigen::SparseMatrix<double>(), false, data);

	Eigen::VectorXd B = Eigen::VectorXd::Zero(data.n, 1);
	igl::min_quad_with_fixed_solve(data, B, UV, Eigen::MatrixXd(), U);
	U.col(0) = -U.col(0);
}


int main(int argc, char* argv[])
{
	// Load input meshes
	Eigen::MatrixXd V, U_tutte, U;
	Eigen::MatrixXi F;

	igl::read_triangle_mesh(
		"../beetle.obj", V, F);

	igl::opengl::glfw::Viewer viewer;

	tutte(V, F, U_tutte);

	// Fit parameterization in unit sphere
	const auto normalize = [](Eigen::MatrixXd& U)
	{
		U.rowwise() -= U.colwise().mean().eval();
		U.array() /=
			(U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff() / 2.0;
	};
	normalize(V);
	normalize(U_tutte);

	bool plot_parameterization = false;
	const auto& update = [&]()
	{
		if (plot_parameterization)
		{
			// Viewer wants 3D coordinates, so pad UVs with column of zeros
			viewer.data().set_vertices(
				(Eigen::MatrixXd(V.rows(), 3) <<
					U.col(0), Eigen::VectorXd::Zero(V.rows()), U.col(1)).finished());
		}
		else
		{
			viewer.data().set_vertices(V);
		}
		viewer.data().compute_normals();
		viewer.data().set_uv(U * 10);
	};
	viewer.callback_key_pressed =
		[&](igl::opengl::glfw::Viewer&, unsigned int key, int)
	{
		switch (key)
		{
		case ' ':
			plot_parameterization ^= 1;
			break;
		case 'c':
			viewer.data().show_texture ^= 1;
			break;
		default:
			return false;
		}
		update();
		return true;
	};

	U = U_tutte;
	viewer.data().set_mesh(V, F);
	Eigen::MatrixXd N;
	igl::per_vertex_normals(V, F, N);
	viewer.data().set_colors(N.array() * 0.5 + 0.5);
	update();
	viewer.data().show_texture = true;
	viewer.data().show_lines = false;

	//Create Edge List
	Eigen::MatrixX2i E;

	igl::edges(F, E);

	//igl::writePLY("../beetle.ply", V, F, E, U_tutte);
	writeObj("../beetleOut.obj", V, F, U_tutte, N);


	viewer.launch();


	return EXIT_SUCCESS;
}

*/