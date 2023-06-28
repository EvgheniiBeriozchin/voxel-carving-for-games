#pragma once
#include <Eigen/Core>


class Camera {
	// TODO: PixelGrid / image for marking while sweeping
public:
	const Eigen::Vector2i& ProjectIntoCameraSpace(Eigen::Vector3d worldPoint) {
		// TODO: project into camera pixel space
	}
	bool IsMarked(Eigen::Vector2i& pixel) {
		// TODO: Check if pixel is unmarked
	}
	void MarkPixel(Eigen::Vector2i& pixel) {
		// TODO: Mark pixel
	}
};