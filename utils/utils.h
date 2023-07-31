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
	std::vector<cv::Point3f> topLeftCorners = {0.01 * cv::Point3f(0.5, 0.5, 0.0), 0.01 * cv::Point3f(4.0, 0.5, 0.0), 0.01 * cv::Point3f(9.0, 0.5, 0.0),
											   0.01 * cv::Point3f(14.0, 0.5, 0.0), 0.01 * cv::Point3f(17.5, 0.5, 0.0), 0.01 * cv::Point3f(17.5, 4.0, 0.0),
											   0.01 * cv::Point3f(17.5, 7.5, 0.0), 0.01 * cv::Point3f(17.5, 11.0, 0.0), 0.01 * cv::Point3f(17.5, 14.5, 0.0),
											   0.01 * cv::Point3f(17.5, 18.0, 0.0), 0.01 * cv::Point3f(17.5, 21.5, 0.0), 0.01 * cv::Point3f(17.5, 25.0, 0.0),
											   0.01 * cv::Point3f(14.0, 25.0, 0.0), 0.01 * cv::Point3f(9.0, 25.0, 0.0), 0.01 * cv::Point3f(4.0, 25.0, 0.0),
											   0.01 * cv::Point3f(0.5, 25.0, 0.0), 0.01 * cv::Point3f(0.5, 21.5, 0.0), 0.01 * cv::Point3f(0.5, 18.0, 0.0),
											   0.01 * cv::Point3f(0.5, 14.5, 0.0), 0.01 * cv::Point3f(0.5, 11.0, 0.0), 0.01 * cv::Point3f(0.5, 7.5, 0.0),
											   0.01 * cv::Point3f(0.5, 4.0, 0.0)};

	for (cv::Point3f topLeftCorner : topLeftCorners)
	{
		std::vector<cv::Point3f> arucoMarker = { topLeftCorner, 
												 topLeftCorner + 0.01 * cv::Point3f(3.0, 0.0, 0.0),
												 topLeftCorner + 0.01 * cv::Point3f(0.0, 3.0, 0.0),
												 topLeftCorner + 0.01 * cv::Point3f(3.0, 3.0, 0.0), };
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
	cv::Size imageSize = cv::Size(1080, 1920);
	std::vector<std::vector<cv::Point3f>> allObjectPoints;
	std::vector<std::vector<cv::Point2f>> allImagePoints;
	int numFrames = video.get(cv::CAP_PROP_FRAME_COUNT);

	for (int i = 0; i < NUM_CALIBRATION_FRAMES; i++)
	{
		if ((i + 1) % 10 == 0)
		{
			std::cout << "Processed " << (i + 1) << " frames" << std::endl;
		}

		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
		std::vector<cv::Point2f> currentImagePoints;
		std::vector<cv::Point3f> currentObjectPoints;
		cv::Mat image;
		video.set(1, i * (numFrames / NUM_CALIBRATION_FRAMES));
		video.retrieve(image);
		imageSize = image.size();

		detector->detectMarkers(image, markerCorners, markerIds, rejectedCandidates);
		detector->refineDetectedMarkers(image, *board, markerCorners, markerIds, rejectedCandidates);
		board->matchImagePoints(
			markerCorners, markerIds,
			currentObjectPoints, currentImagePoints
		);

		allObjectPoints.push_back(currentObjectPoints);
		allImagePoints.push_back(currentImagePoints);
	}



	cv::Mat localCameraMatrix, localDistortionCoefficients;
	std::vector<cv::Mat> rvecs, tvecs;
	int calibrationFlags = cv::CALIB_USE_INTRINSIC_GUESS;
	float cm_f[9] = { 1823.3715871387003, 0, 540,
					  0, 1823.3715871387003, 960,
					  0, 0, 1 };
	localCameraMatrix = cv::Mat(3, 3, CV_32F, cm_f);
	double repError = cv::calibrateCamera(
		allObjectPoints, allImagePoints, imageSize,
		localCameraMatrix, localDistortionCoefficients, rvecs, tvecs, cv::noArray(),
		cv::noArray(), cv::noArray(), calibrationFlags
	);

	for (int i = 0; i < NUM_CALIBRATION_FRAMES; i++)
	{
		cv::Mat imageCopy;
		if ((i + 1) % 10 == 0)
		{
			std::cout << "Processed " << (i + 1) << " frames" << std::endl;
		}

		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

		cv::Mat image, currentObjectPoints, currentImagePoints;
		video.set(1, i * (numFrames / NUM_CALIBRATION_FRAMES));
		video.retrieve(image);
		image.copyTo(imageCopy);
		cv::drawFrameAxes(imageCopy, localCameraMatrix, localDistortionCoefficients, rvecs[i], tvecs[i], 0.1);

		Camera camera = Camera(image, localCameraMatrix);
		camera.pose = camera.estimateCameraPose(detector, board, localCameraMatrix, localDistortionCoefficients);
		for (int j = 0; j < allImagePoints[i].size(); j++)
		{
			cv::Point2d pixelPos = allImagePoints[i][j];
			cv::circle(imageCopy, cv::Point(pixelPos.x, pixelPos.y), 5, cv::Vec3b(255, 0, 0), 5);

			Eigen::Vector3d pos = Eigen::Vector3d(allObjectPoints[i][j].x, allObjectPoints[i][j].y, allObjectPoints[i][j].z);
			Eigen::Vector2i pixelPos2 = camera.ProjectIntoCameraSpace(pos);
			cv::circle(imageCopy, cv::Point(pixelPos2.x(), pixelPos2.y()), 5, cv::Vec3b(0, 255, 0), 5);
		}

		cv::imwrite("./calibrationResults/frame-" + std::to_string(i) + ".jpg", imageCopy);
	}

	/*
	float cm_f[9] = { 1823.3715871387003, 0, 540,
					  0, 1823.3715871387003, 960,
					  0, 0, 1};
	cv::Mat cm = cv::Mat(3, 3, CV_32F, cm_f);
	float dc_f[5] = { 0.018327194204981752, 0, 0, 0, 0 };
	cv::Mat dc = cv::Mat(1, 5, CV_32F, cm_f);

	localCameraMatrix = cm;
	localDistortionCoefficients = dc;

	
	float cm_f[9] = { 1856.296491838788, 0, 519.8483668408671,
					  0, 1858.04034342115, 955.0267490933753,
					  0, 0, 1 }; 
	float cm_f[9] = {3461.194051903984, 0, 539.2599944955601,
	0, 3215.14225399305, 903.9439469919649,
	0, 0, 1 };
	cv::Mat cm = cv::Mat(3, 3, CV_32F, cm_f);

	//float dc_f[5] = { -0.0641685889523109, 1.33098094466823, -0.003969580929144683, -0.0007438596026286802, -3.428214031198218 };
	float dc_f[5] = { -24.29358258111102, 936.8014431638936, 0.02815050387029168, -0.05042948592282346, -11468.8350942283 };
	cv::Mat dc = cv::Mat(1, 5, CV_32F, cm_f);

	std::cout << cm << std::endl;
	std::cout << dc << std::endl;

	localCameraMatrix = cm;
	localDistortionCoefficients = dc;
	*/
	/*
	cv::Mat image, outputImage;
	video.retrieve(image);
	cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(localCameraMatrix, localDistortionCoefficients, imageSize, 1, imageSize, 0);

	// Method 1 to undistort the image
	// cv::undistort(image, outputImage, newCameraMatrix, localDistortionCoefficients, newCameraMatrix);

	// Method 2 to undistort the image
	cv::Mat map1, map2;
	cv::initUndistortRectifyMap(localCameraMatrix, localDistortionCoefficients, cv::Mat(), newCameraMatrix, imageSize, CV_16SC2, map1, map2);
	cv::remap(image, outputImage, map1, map2, cv::INTER_LINEAR);

	//Displaying the undistorted image
	cv::imshow("undistorted image", outputImage);
	cv::waitKey(0);;

	*cameraMatrix = newCameraMatrix;
	*/

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