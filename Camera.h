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
	std::vector<cv::Point3d> objectPoints;

	Camera(cv::Mat image, const cv::Mat cameraMatrix)
	{
		cv::Size size = image.size();

		frame = image;
		markingFrame = cv::Mat(size, CV_8U, cv::Scalar(0));
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

	//const Eigen::Vector2i ProjectIntoCameraSpace(Eigen::Vector3d worldPoint) {
	//	Eigen::Vector4d worldPoint4 = Eigen::Vector4d(worldPoint[0], worldPoint[1], worldPoint[2], 1.0f);
	//	// Convert world point to homogeneous coordinates
	//	cv::Mat worldCoordinateHomogeneous(4, 1, CV_64F);
	//	cv::eigen2cv(worldPoint4, worldCoordinateHomogeneous);
	//	cv::Mat cameraCoordinateHomogeneous = cameraTransform * worldCoordinateHomogeneous;
	//	cv::Mat projectedPoint = cameraMatrix * cameraCoordinateHomogeneous(cv::Rect(0, 0, 1, 3));
	//	float xNormalized = projectedPoint.at<double>(0) / projectedPoint.at<double>(2);
	//	float yNormalized = projectedPoint.at<double>(1) / projectedPoint.at<double>(2);
	//	return Eigen::Vector2i(xNormalized, yNormalized);
	//}

	bool IsMarked(Eigen::Vector2i& pixel) {
		return markingFrame.at<uchar>(pixel.y(), pixel.x()) == 255;
	}

	void MarkPixel(Eigen::Vector2i& pixel) {
		markingFrame.at<uchar>(pixel.y(), pixel.x()) = 255;
	}

	Eigen::Matrix4d estimateCameraPose(cv::aruco::ArucoDetector *detector, cv::aruco::Board *board, cv::Mat cameraMatrix, cv::Mat distortionCoefficients)
	{
		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
		cv::Mat objectPoints, imagePoints;
		cv::Vec3d rotationVector, translationVector;

		detector->detectMarkers(frame, markerCorners, markerIds, rejectedCandidates);
		detector->refineDetectedMarkers(frame, *board, markerCorners, markerIds, rejectedCandidates);
		board->matchImagePoints(markerCorners, markerIds, objectPoints, imagePoints);
		cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distortionCoefficients, rotationVector, translationVector);
		cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distortionCoefficients, rotationVector, translationVector);
		
		
		this->objectPoints = objectPoints;
		
		Eigen::Vector3d position = Eigen::Vector3d(translationVector[0], translationVector[1], translationVector[2]);
		cv::Mat cvRotationMatrix;
		cv::Rodrigues(rotationVector, cvRotationMatrix);
		cvRotationMatrix = cvRotationMatrix.t();

		Eigen::Matrix4d pose;
		Eigen::Matrix3d rotationMatrix;
		cv::cv2eigen(cvRotationMatrix, rotationMatrix);
		position = rotationMatrix * position;
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
		//cv::adaptiveThreshold(grayScaleFrame, grayScaleFrame, maxValue, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);
	}
};