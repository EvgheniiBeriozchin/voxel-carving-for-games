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
	std::vector<std::vector<cv::Point3f>> arucoMarkers;
	std::vector<cv::Point3f> topLeftCorners = {cv::Point3f(0.5, 0.5, 0.0), cv::Point3f(4.0, 0.5, 0.0), cv::Point3f(9.0, 0.5, 0.0), 
											   cv::Point3f(14.0, 0.5, 0.0), cv::Point3f(17.5, 0.5, 0.0), cv::Point3f(17.5, 4.0, 0.0),
											   cv::Point3f(17.5, 7.5, 0.0), cv::Point3f(17.5, 11.0, 0.0), cv::Point3f(17.5, 14.5, 0.0),
											   cv::Point3f(17.5, 18.0, 0.0), cv::Point3f(17.5, 21.5, 0.0), cv::Point3f(17.5, 25.0, 0.0),
											   cv::Point3f(14.0, 25.0, 0.0), cv::Point3f(9.0, 25.0, 0.0), cv::Point3f(4.0, 25.0, 0.0),
											   cv::Point3f(0.5, 25.0, 0.0), cv::Point3f(0.5, 21.5, 0.0), cv::Point3f(0.5, 18.0, 0.0),
											   cv::Point3f(0.5, 14.5, 0.0), cv::Point3f(0.5, 11.0, 0.0), cv::Point3f(0.5, 7.5, 0.0),
											   cv::Point3f(0.5, 4.0, 0.0)};

	for (cv::Point3f topLeftCorner : topLeftCorners)
	{
		std::vector<cv::Point3f> arucoMarker = { topLeftCorner, 
												 topLeftCorner + cv::Point3f(3.0, 0.0, 0.0), 
												 topLeftCorner + cv::Point3f(0.0, 3.0, 0.0), 
												 topLeftCorner + cv::Point3f(3.0, 3.0, 0.0), };
		arucoMarkers.push_back(arucoMarker);
	}

	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
	std::vector<int> markerIds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

	return new cv::aruco::Board(arucoMarkers, dictionary, markerIds);
}


const int NUM_CALIBRATION_FRAMES = 20;

void calibrateCamera(cv::VideoCapture video, cv::aruco::ArucoDetector* detector, cv::aruco::Board* board, 
					 cv::Mat* cameraMatrix, cv::Mat* distortionCoefficients)
{
	std::cout << "Calibrating camera" << std::endl;
	cv::Size imageSize = cv::Size(256, 256);
	std::vector<cv::Mat> allObjectPoints, allImagePoints;
	int numFrames = video.get(cv::CAP_PROP_FRAME_COUNT);

	for (int i = 0; i < NUM_CALIBRATION_FRAMES; i++)
	{
		if ((i + 1) % 10 == 0)
		{
			std::cout << "Processed " << (i + 1) << " frames" << std::endl;
		}

		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

		cv::Mat image, currentObjectPoints, currentImagePoints;
		video.set(1, i * (numFrames / NUM_CALIBRATION_FRAMES));
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

bool videoExists(cv::VideoCapture video)
{
	if (!video.isOpened())
	{
		std::cout << "Couldn't open video" << std::endl;
		return false;
	}
	else
	{
		std::cout << "Video has " << video.get(cv::CAP_PROP_FRAME_COUNT) << " frames" << std::endl;
		return true;
	}
}