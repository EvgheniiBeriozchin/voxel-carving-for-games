#define RunOpenCVTest 1

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

int main() {
	if (RunOpenCVTest) {
		namedWindow("Test_window", WINDOW_AUTOSIZE);
		while (waitKey(100) <= 0) {

		}
	}
	return 0;
}
