#pragma once
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>

struct Pose {
	Eigen::Vector3f position;
	Eigen::Vector3f rotation;
};

bool t=false;
class Camera {
public:
	cv::Mat frame;
	cv::Mat grayScaleFrame;
	cv::Mat markingFrame;
	Pose pose;
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
		Eigen::Vector3d screenSpaceIntermediate = instrinsicMatrix * worldPoint;

		return Eigen::Vector2i(screenSpaceIntermediate.x() / screenSpaceIntermediate.z(),
							   screenSpaceIntermediate.y() / screenSpaceIntermediate.z());
	}

	bool IsMarked(Eigen::Vector2i& pixel) {
		return markingFrame.at<uchar>(pixel.x(), pixel.y(), 0);
	}

	void MarkPixel(Eigen::Vector2i& pixel) {
		markingFrame.at<uchar>(pixel.x(), pixel.y(), 0) = 1;
	}
	Pose estimateCameraPose(cv::aruco::ArucoDetector *detector, cv::aruco::Board *board, cv::Mat cameraMatrix, cv::Mat distanceCoefficients)
	{
		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
		cv::Mat objectPoints, imagePoints;
		cv::Vec3d rotationVector, translationVector;

		detector->detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);
		board->matchImagePoints(markerCorners, markerIds, objectPoints, imagePoints);
		cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distanceCoefficients, rotationVector, translationVector);

		Pose pose;
		pose.position = Eigen::Vector3f(translationVector[0], translationVector[1], translationVector[2]);
		pose.rotation = Eigen::Vector3f(rotationVector[0], rotationVector[1], rotationVector[2]);

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