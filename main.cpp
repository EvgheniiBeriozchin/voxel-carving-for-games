#include <stdio.h>
#include <stdlib.h>

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
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/writePLY.h>
#include <igl/edges.h>
#include <export.h>


#include <voxel/SimpleMesh.h>
#include <voxel/MarchingCubes.h>


#define RUN_CAMERA_CALIBRATION 1
#define RUN_POSE_ESTIMATION_TEST 0
#define RUN_VOXEL_GRID_TEST 0
#define RUN_VOXEL_CARVING 1
#define RUN_CAMERA_ESTIMATION_EXPORT 0
#define RUN_TUTTE_EMBEDDING 0
#define EXPORT_TEXTURED_MESH 1


const int NUM_PROCESSED_FRAMES = 25;
const std::string CALIBRATION_VIDEO_NAME = "../PepperMill_NaturalLight.mp4";
const std::string RECONSTRUCTION_VIDEO_NAME = "../PepperMill_NaturalLight.mp4";
//const std::string RECONSTRUCTION_VIDEO_NAME = "../Box_NaturalLight.mp4";
const std::string voxeTestFilenameTarget = std::string("voxelGrid.off");
const std::string uvTestingInput = "../bunny.off";

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

		int numFrames = reconstructionVideo.get(cv::CAP_PROP_FRAME_COUNT);
		for (int i = 0; i < NUM_PROCESSED_FRAMES; i++)
		{
			reconstructionVideo.set(1, i * (numFrames / NUM_PROCESSED_FRAMES));
			cv::Mat image;
			reconstructionVideo.retrieve(image);

			Camera frame = Camera(image, cameraMatrix);
			Eigen::Matrix4d currentPose = frame.estimateCameraPose(&detector, board, cameraMatrix, distanceCoefficients);

			std::cout << "Camera pose: " << currentPose << std::endl;
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


		Eigen::Vector3d boardCenter = 0.01 * Eigen::Vector3d(10.5, 14.25, 0.0);
		Eigen::Vector3d gridOrigin = 0.01 * Eigen::Vector3d(3.5, 3.5, 0);
		// Real dimension cm
		double xSizeCM = 14;
		double ySizeCM = 21.5;
		double zSizeCM = 10;
		// VoxelDimension

		double voxelPerCM = 2;
		double xSizeVX = xSizeCM * voxelPerCM;
		double ySizeVX = ySizeCM * voxelPerCM;
		double zSizeVX = zSizeCM * voxelPerCM;
		double voxelSize = 0.01 / voxelPerCM;
		auto grid = VoxelGrid::CreateFilledVoxelGrid(gridOrigin, Eigen::Vector3i(xSizeVX, ySizeVX, zSizeVX), voxelSize);
		//auto grid = VoxelGrid::CreateFilledVoxelGrid(gridOrigin, Eigen::Vector3i(3, 3, 5), 0.05);

		std::cout << "Preparing frames for voxel carving" << std::endl;
		for (int i = 0; i < NUM_PROCESSED_FRAMES; i++)
		{
			if ((i + 1) % 10 == 0)
			{
				std::cout << "Processed " << (i + 1) << " frames" << std::endl;
			}
			cv::Mat image;
			reconstructionVideo.set(1, i * (numFrames / NUM_PROCESSED_FRAMES));
			reconstructionVideo.retrieve(image);

			Camera cam = Camera(image, cameraMatrix);
			cam.pose = cam.estimateCameraPose(&detector, board, cameraMatrix, distanceCoefficients);
			cameraFrames.push_back(cam);
		}



		// Camera To markers output
		if (RUN_CAMERA_ESTIMATION_EXPORT) {
			int ci = 0;
			std::vector<std::vector<Eigen::Vector3d>> points;
			points.push_back(std::vector<Eigen::Vector3d>());
			std::vector<Eigen::Vector3d> colors;
			colors.push_back(Eigen::Vector3d(0, 255, 255));
			for (int i = 0; i < cameraFrames.size(); i++)
			{
				auto pos = cameraFrames[i].pose.block<3, 1>(0, 3);
				points[ci].push_back(pos);
			}
			// x axis
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(255, 0, 0));
			for (int i = -100; i < 100; i += 2) {
				points[ci].push_back(0.01 * Eigen::Vector3d(i, 0, 0));
			}
			// y axis
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(0, 255, 0));
			for (int i = -100; i < 100; i += 2) {
				points[ci].push_back(0.01 * Eigen::Vector3d(0, i, 0));
			}
			// z axis
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(0, 0, 255));
			for (int i = 0; i < 100; i += 2) {
				points[ci].push_back(0.01 * Eigen::Vector3d(0, 0, i));
			}
			// marker setup
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(0, 0, 0));
			points[ci].push_back(boardCenter);
			for (int i = 0; i < boardCenter.x() * 2; i++) {
				points[ci].push_back(0.01 * Eigen::Vector3d(i, 0, 0));
				points[ci].push_back(Eigen::Vector3d(0.01 * i, boardCenter.y() * 2, 0));
			}
			for (int i = 0; i < boardCenter.y() * 2; i++) {
				points[ci].push_back(0.01 * Eigen::Vector3d(0, i, 0));
				points[ci].push_back(Eigen::Vector3d(boardCenter.x() * 2, 0.01 * i, 0));
			}
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(255, 255, 255));
			for (int i = 0; i < cameraFrames.size(); i++)
			{
				for (int j = 0; j < cameraFrames[i].objectPoints.size(); j++) {
					points[ci].push_back(Eigen::Vector3d(
						cameraFrames[i].objectPoints[j].x,
						cameraFrames[i].objectPoints[j].y,
						cameraFrames[i].objectPoints[j].z
					));
				}
			}

			VoxelGridExporter::ExportToPLY("cameraPoses.ply", points, colors);
			VoxelGridExporter::ExportToOFF("voxelGrid_cameraPoses.off", grid);
		}
		int count = 0;
		for (int i = 0; i < cameraFrames.size(); i++)
		{
			Eigen::Vector2i projection = cameraFrames[i].ProjectIntoCameraSpace(Eigen::Vector3d(0, 0, 0));
			if (projection.x() >= 0 && projection.x() < cameraFrames[i].frame.size().width
				&& projection.y() >= 0 && projection.y() < cameraFrames[i].frame.size().height)
			{
				count++;
				std::cout << "World center location: " << projection << std::endl;
			}
		}
		std::cout << "Center points within frame: " << count << std::endl;

		// write image with projected grid positions
		cv::Mat tf;
		int index = 0;

		for (/*auto cameraFrame: cameraFrames*/int i = 0; i < 1; i++)
		{
			//if (index > 10)
			//	break;
			auto cameraFrame = cameraFrames[i];
			cameraFrame.frame.copyTo(tf);
			for each (auto v in grid.GetBoundaryVoxels())
			{
				Eigen::Vector3d v2 = grid.GetVoxelCenter(v);
				auto pixelPos = cameraFrame.ProjectIntoCameraSpace(v2);
				if (pixelPos.x() >= tf.cols || pixelPos.y() >= tf.rows || pixelPos.x() < 0 || pixelPos.y() < 0)
					continue;
				cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(0, 0, 255), 5);
			}
			auto pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0, 0, 0));
			//std::cout << "00 pixel: " << pixelPos << std::endl;
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(255, 255, 0), 5);
			//std::cout << "010 pixel: " << pixelPos << std::endl;
			pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0.1, 0, 0));
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(0, 0, 255), 5);
			//std::cout << "001 pixel: " << pixelPos << std::endl;
			pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0, 0.1, 0));
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(0, 255, 0), 5);
			//std::cout << "0101 pixel: " << pixelPos << std::endl;
			pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0, 0, 0.1));
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(255, 0, 0), 5);
			cv::imwrite("camera_" + std::to_string(index++) + "_gray.png", tf);
		}
		std::cout << "Running voxel carving" << std::endl;
		SpaceCarver::MultiSweep(grid, cameraFrames);
		VoxelGridExporter::ExportToOFF(voxeTestFilenameTarget, grid);

		SimpleMesh mesh;
		CreateMesh(&grid, &mesh);

		if (EXPORT_TEXTURED_MESH) {
			// Load input meshes
			Eigen::MatrixXd V, U, N;
			Eigen::MatrixXi F, C;

			mesh.GetMeshData(V, F, C);

			// Debug
			// igl::read_triangle_mesh(uvTestingInput, V, F);

			TutteEmbedder::GenerateUvMapping(V, F, U, N);

			//C is in [0,255] rgb, export and render need [0,1] rgb
			Eigen::MatrixXd colors = C.cast<double>() / 255.0;

			MeshExport::WriteObj("mesh", V, F, U, N, colors);

			MeshExport::RenderTexture("mesh", U, F, colors);

			return 0;
		}

		mesh.WriteMesh("mesh.off");
	}

	if (RUN_TUTTE_EMBEDDING) {
		// Load input meshes
		Eigen::MatrixXd V, U, N;
		Eigen::MatrixXi F;

		igl::read_triangle_mesh(uvTestingInput, V, F);

		TutteEmbedder::GenerateUvMapping(V, F, U, N);

		Eigen::MatrixXd colors = Eigen::MatrixXd::Random(V.rows(), 3);
		colors = (colors + Eigen::MatrixXd::Constant(V.rows(), 3, 1.)) / 2.;

		MeshExport::WriteObj("mesh", V, F, U, N, colors);

		MeshExport::RenderTexture("mesh", U, F, colors);
	}

	return 0;
}