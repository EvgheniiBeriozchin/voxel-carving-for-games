#pragma once
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

struct Pose {
	Eigen::Vector3f position;
	Eigen::Vector3f rotation;
};

cv::aruco::ArucoDetector createDetector()
{
	cv::aruco::DetectorParameters parameters = cv::aruco::DetectorParameters();
	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
	cv::aruco::ArucoDetector detector(dictionary, parameters);

	return detector;
}

cv::aruco::Board* createBoard()
{
	return new cv::aruco::Board();
}

void calibrateCamera(cv::VideoCapture video, cv::aruco::ArucoDetector *detector, cv::aruco::Board *board,
						cv::Mat *cameraMatrix, cv::Mat *distanceCoefficients)
{
	cv::Size imageSize = cv::Size(256, 256);
	std::vector<cv::Mat> allObjectPoints, allImagePoints;

	while (video.grab())
	{
		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

		cv::Mat image, currentObjectPoints, currentImagePoints;
		video.retrieve(image);

		detector->detectMarkers(image, markerCorners, markerIds, rejectedCandidates);
		board->matchImagePoints(
			markerCorners, markerIds,
			currentObjectPoints, currentImagePoints
		);

		allObjectPoints.push_back(currentObjectPoints);
		allImagePoints.push_back(currentImagePoints);
	}



	cv::Mat localCameraMatrix, localDistanceCoefficients;
	std::vector<cv::Mat> rvecs, tvecs;
	int calibrationFlags = 0;
	double repError = calibrateCamera(
		allObjectPoints, allImagePoints, imageSize,
		localCameraMatrix, localDistanceCoefficients, rvecs, tvecs, cv::noArray(),
		cv::noArray(), cv::noArray(), calibrationFlags
	);
	
	*cameraMatrix = localCameraMatrix;
	*distanceCoefficients = localDistanceCoefficients;
}

Pose estimateCameraPose(cv::Mat inputImage, cv::aruco::ArucoDetector *detector, cv::aruco::Board *board,
									cv::Mat cameraMatrix, cv::Mat distanceCoefficients)
{
	std::vector<int> markerIds;
	std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
	cv::Mat objectPoints, imagePoints;
	cv::Vec3d rotationVector, translationVector;

	detector->detectMarkers(inputImage, markerCorners, markerIds, rejectedCandidates);
	board->matchImagePoints(markerCorners, markerIds, objectPoints, imagePoints);
	cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distanceCoefficients, rotationVector, translationVector);

	Pose pose;
	pose.position = Eigen::Vector3f(translationVector[0], translationVector[1], translationVector[2]);
	pose.rotation = Eigen::Vector3f(rotationVector[0], rotationVector[1], rotationVector[2]);

	return pose;
}

