#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

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

void calibrateCamera(cv::VideoCapture video, cv::aruco::ArucoDetector* detector, cv::aruco::Board* board, 
					 cv::Mat* cameraMatrix, cv::Mat* distortionCoefficients)
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



	cv::Mat localCameraMatrix, localDistortionCoefficients;
	std::vector<cv::Mat> rvecs, tvecs;
	int calibrationFlags = 0;
	double repError = cv::calibrateCamera(
		allObjectPoints, allImagePoints, imageSize,
		localCameraMatrix, localDistortionCoefficients, rvecs, tvecs, cv::noArray(),
		cv::noArray(), cv::noArray(), calibrationFlags
	);

	*cameraMatrix = localCameraMatrix;
	*distortionCoefficients = localDistortionCoefficients;
}