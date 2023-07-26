#include <iostream>
#include <opencv2/opencv.hpp>

#include "voxel/VoxelGrid.h"
#include "voxel/VoxelGridExporter.h"
#include "voxel/SpaceCarver.h"
#include "Camera.h"
#include "utils/utils.h"
#include "TutteEmbedding.h"

#include <igl/read_triangle_mesh.h>

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
		Eigen::MatrixXd V, U_tutte, U;
		Eigen::MatrixXi F;
		Eigen::MatrixXd N;
		igl::readOBJ(uvTestingInput, V, F);
		TutteEmbedder::GenerateUvMapping(V, F, U, N);

		igl::opengl::glfw::Viewer viewer;

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

		viewer.data().set_mesh(V, F);
		viewer.data().set_colors(N.array() * 0.5 + 0.5);
		update();
		viewer.data().show_texture = true;
		viewer.data().show_lines = false;
		viewer.launch();
	}
	
	return 0;
}
 