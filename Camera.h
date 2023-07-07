#pragma once
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>

bool t=false;
class Camera {
public:
	cv::Mat frame;
	cv::Mat grayScaleFrame;
	cv::Mat markingFrame;
	Eigen::Matrix4d pose;
	Eigen::Matrix3d instrinsicMatrix;

	Camera(cv::Mat image, const cv::Mat cameraMatrix)
	{
		cv::Size size = image.size();

		frame = image;
		markingFrame = cv::Mat(size, CV_8UC1);
		grayScaleFrame = cv::Mat(size, CV_8UC1);
		cv::cv2eigen(cameraMatrix, instrinsicMatrix);
		
		prepareImage();
	}

	const Eigen::Vector2i& ProjectIntoCameraSpace(Eigen::Vector3d worldPoint) {
		Eigen::Vector4d worldPoint4 = Eigen::Vector4d(worldPoint[0], worldPoint[1], worldPoint[2], 1.0f);
		Eigen::Matrix<double, 3, 4> reshapingMatrix = Eigen::Matrix<double, 3, 4>::Identity();
		Eigen::Vector3d screenSpaceIntermediate = instrinsicMatrix * reshapingMatrix * pose * worldPoint4;

		return Eigen::Vector2i(screenSpaceIntermediate.x() / screenSpaceIntermediate.z(),
							   screenSpaceIntermediate.y() / screenSpaceIntermediate.z());
	}

	bool IsMarked(Eigen::Vector2i& pixel) {
		return markingFrame.at<uchar>(pixel.x(), pixel.y());
	}

	void MarkPixel(Eigen::Vector2i& pixel) {
		markingFrame.at<uchar>(pixel.x(), pixel.y()) = 1;
	}

	Eigen::Matrix4d estimateCameraPose(cv::aruco::ArucoDetector *detector, cv::aruco::Board *board, cv::Mat cameraMatrix, cv::Mat distanceCoefficients)
	{
		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
		cv::Mat objectPoints, imagePoints;
		cv::Vec3d rotationVector, translationVector;

		detector->detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);
		detector->refineDetectedMarkers(frame, *board, markerCorners, markerIds, rejectedCandidates);
		board->matchImagePoints(markerCorners, markerIds, objectPoints, imagePoints);
		cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distanceCoefficients, rotationVector, translationVector);

		Eigen::Vector3d position = Eigen::Vector3d(translationVector[0], translationVector[1], translationVector[2]);
		cv::Mat cvRotationMatrix;
		cv::Rodrigues(rotationVector, cvRotationMatrix);
		Eigen::Matrix4d pose;
		Eigen::Matrix3d rotationMatrix;
		cv::cv2eigen(cvRotationMatrix, rotationMatrix);
		pose.block(0, 0, 3, 3) = rotationMatrix;
		pose.block(0, 3, 3, 1) = position;

		return pose;
	}

private:
	
	void prepareImage() {
		cv::cvtColor(frame, grayScaleFrame, cv::COLOR_BGR2GRAY);
		double maxValue = 255;
		int blockSize = 11;    // Size of the neighborhood for thresholding (should be odd)
		double C = 2;
		cv::adaptiveThreshold(grayScaleFrame, grayScaleFrame, maxValue, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
	}
};