#include <iostream>
#include <opencv2/opencv.hpp>

#include "Camera.h"

#define RUN_CAMERA_CALIBRATION 1
#define RUN_POSE_ESTIMATION_TEST 2

int main() {
	cv::Mat cameraMatrix, distanceCoefficients;
	cv::VideoCapture calibrationVideo, reconstructionVideo;
	cv::aruco::ArucoDetector detector = createDetector();
	cv::aruco::Board *board = createBoard();


	if (RUN_CAMERA_CALIBRATION)
	{
		calibrateCamera(calibrationVideo, &detector, board, &cameraMatrix, &distanceCoefficients);
	}

	if (RUN_POSE_ESTIMATION_TEST)
	{
		while (reconstructionVideo.grab())
		{
			cv::Mat image;
			reconstructionVideo.retrieve(image);

			Pose currentPose = estimateCameraPose(image, &detector, board, cameraMatrix, distanceCoefficients);

			std::cout << "Camera translation vector: " << currentPose.position.x() << ", "
				<< currentPose.position.y() << ", " << currentPose.position.z() << std::endl;
			std::cout << "Camera rotation vector: " << currentPose.rotation.x() << ", "
				<< currentPose.rotation.y() << ", " << currentPose.rotation.z() << std::endl;
		}
	}

	return 0;
}
 