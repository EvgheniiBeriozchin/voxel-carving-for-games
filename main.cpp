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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

void Barycentric(Eigen::Vector3d& p, Eigen::Vector3d& a, Eigen::Vector3d& b, Eigen::Vector3d& c, float &u, float &v, float &w) {
	Eigen::Vector3d v0 = b - a, v1 = c - a, v2 = p - a;
	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = d00 * d11 - d01 * d01;
	v = (d11 * d20 - d01 * d21) / denom;
	w = (d00 * d21 - d01 * d20) / denom;
	u = 1.0f - v - w;
}

void writeTextureFile(std::string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& VertexColors) {


	std::cout << "Writing texture file" << std::endl;

	int width = 1280;
	int height = 720;

	Eigen::MatrixXd Colors(width * 3, height );

	std::cout << width * height * F.rows() << " iterations needed" << std::endl;

	for (unsigned int x = 0; x < width; x + 3) {
		std::cout << "Row " << x << " started" << std::endl;
		for (unsigned int y = 0; y < height; y++) {
			//get barycentric coordinates for every triangle
			for (unsigned int k = 0; k < F.rows(); k++) {

				Eigen::Vector3d p1 = V.row(F(k, 0));
				Eigen::Vector3d p2 = V.row(F(k, 1));
				Eigen::Vector3d p3 = V.row(F(k, 2));

				Eigen::Vector3d color1 = VertexColors.row(F(k, 0));
				Eigen::Vector3d color2 = VertexColors.row(F(k, 1));
				Eigen::Vector3d color3 = VertexColors.row(F(k, 2));

				float x_u = ((float)x / 3) / (float)width;
				float y_v = y / (float)height;

				float u, v, w;

				Barycentric(Eigen::Vector3d(x_u, y_v, 0), p1, p2, p3, u, v, w);

				if (u < 0 || v < 0 || w < 0) {
					continue;
				}

				Colors(x, y) = u * color1(0) + v * color2(0) + w * color3(0);
				Colors(x + 1, y) = u * color1(1) + v * color2(1) + w * color3(1);
				Colors(x + 2, y) = u * color1(2) + v * color2(2) + w * color3(2);
			}
		}
	}

	std::string texFileName = filename.substr(0, filename.find_last_of(".")) + ".png";

	stbi_write_png(texFileName.c_str(), width, height, 3, Colors.data(), width * 3);
}

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

	//Create .mtl file reference
	std::string mtlFileName = filename.substr(0, filename.find_last_of(".")) + ".mtl";
	myfile << "mtllib " << mtlFileName << std::endl;

	myfile.close();

	//Create .mtl file
	myfile.open(mtlFileName);
	myfile << "newmtl material_0" << std::endl;
	myfile << "Ka 1.000000 1.000000 1.000000" << std::endl;
	myfile << "Kd 1.000000 1.000000 1.000000" << std::endl;
	myfile << "Ks 0.000000 0.000000 0.000000" << std::endl;
	myfile << "d 1.0" << std::endl;
	myfile << "illum 2" << std::endl;
	myfile << "Ns 0.000000" << std::endl;
	myfile << "map_Kd " << filename.substr(0, filename.find_last_of(".")) << ".png" << std::endl;
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

		Eigen::MatrixXd colors = Eigen::MatrixXd::Random(V.rows(), 3);

		writeObj("../beetleOut.obj", V, F, U_tutte, N);
		writeTextureFile("../beetleOut.png", V, F, colors);


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