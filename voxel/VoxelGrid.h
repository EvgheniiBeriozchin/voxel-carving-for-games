#pragma once

#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../Camera.h"

struct Voxel {
	int value;
	std::vector<Camera> cameras;
	Voxel() :value(0) {}
	Voxel(int val) : value(val) {}
};

class VoxelGrid {
public:
	VoxelGrid(double voxelSize, const Eigen::Vector3d& origin) : VoxelGrid(voxelSize, origin, Eigen::Vector3i().Zero()) {
	}
    VoxelGrid(double voxelSize, const Eigen::Vector3d& origin, const Eigen::Vector3i& dimensions) :
        voxelSize_(voxelSize),
        origin_(origin),
        voxelCenterOffset_(Eigen::Vector3d(voxelSize / 2.0, voxelSize / 2.0, -voxelSize / 2.0)),
        dimensions_(dimensions) {
        voxels_ = std::vector<Voxel>(dimensions.x() * dimensions.y() * dimensions.z());
    }

	const double GetVoxelSize() {
		return voxelSize_;
	}

	void SetDimensions(const Eigen::Vector3i& dimensions) {
		dimensions_ = dimensions;
		voxels_.resize(dimensions_.x() * dimensions_.y() * dimensions_.z());
	}
	const Eigen::Vector3i& GetDimensions() {
		return dimensions_;
	}

	void AddVoxel(const Eigen::Vector3i& position, const Voxel& voxel) {
		int index = GetIndex(position);
		voxels_[index] = voxel;
	}

	void RemoveVoxel(const Eigen::Vector3i& position) {
		int index = GetIndex(position);
		voxels_[index] = Voxel(0);
	}

	Voxel& GetVoxel(const Eigen::Vector3i& position) {
		int index = GetIndex(position);
		if(index < 0 || index >= voxels_.size())
			return Voxel(0);
		return voxels_[index];
	}

	size_t GetVoxelCount() {
		return voxels_.size();
	}

	const std::vector<Eigen::Vector3d> GetSetVoxelCenterPoints() {
		std::vector<Eigen::Vector3d> positions;
		for (int x = 0; x < dimensions_.x(); ++x) {
			for (int y = 0; y < dimensions_.y(); ++y) {
				for (int z = 0; z < dimensions_.z(); ++z) {
					const Eigen::Vector3i voxelPosition(x, y, z);
					const Voxel& voxel = GetVoxel(voxelPosition);
					if (voxel.value == 1) {
						const Eigen::Vector3d centerPosition = GetVoxelCenter(voxelPosition);
						positions.push_back(centerPosition);
					}
				}
			}
		}

		return positions;
	}
	const std::vector<Eigen::Vector3d> GetVoxelCenterPoints(const std::vector<Eigen::Vector3i> voxels) {
		std::vector<Eigen::Vector3d> positions;

		for each (auto pos in voxels)
		{
			const Eigen::Vector3d centerPosition = GetVoxelCenter(pos);
			positions.push_back(centerPosition);
		}
		return positions;
	}

	const std::vector<Eigen::Vector3i> GetSetVoxels() {
		std::vector<Eigen::Vector3i> voxelsPositions;
		for (int x = 0; x < dimensions_.x(); ++x) {
			for (int y = 0; y < dimensions_.y(); ++y) {
				for (int z = 0; z < dimensions_.z(); ++z) {
					const Eigen::Vector3i voxelPosition(x, y, z);
					Voxel& voxel = GetVoxel(voxelPosition);
					if (voxel.value == 1) {
						voxelsPositions.push_back(voxelPosition);
					}
				}
			}
		}
		return voxelsPositions;
	}

	const Eigen::Vector3d GetVoxelPositionFromIndex(int index) {
		const int z = index / (dimensions_.x() * dimensions_.y());
		index -= z * dimensions_.x() * dimensions_.y();
		const int y = index / dimensions_.x();
		const int x = index % dimensions_.x();
		return Eigen::Vector3d(x, y, z);
	}

	const Eigen::Vector3d IndexToPosition(const Eigen::Vector3i& index) {
		return Eigen::Vector3d(index.x(), index.y(), index.z()) * voxelSize_ + origin_;
	}

	const Eigen::Vector3d IndexToPosition(const int & x, const int& y, const int& z) {
		return Eigen::Vector3d(x, y, z) * voxelSize_ + origin_;
	}

	const std::vector<Eigen::Vector3i> GetBoundaryVoxels() {
		std::vector<Eigen::Vector3i> boundaryVoxels;
		for (int x = 0; x < dimensions_.x(); ++x) {
			for (int y = 0; y < dimensions_.y(); ++y) {
				for (int z = 0; z < dimensions_.z(); ++z) {
					bool isBoundaryVoxel = false;
					if (GetVoxel(Eigen::Vector3i(x, y, z)).value == 0)
						continue;
					for (int dx = -1; dx <= 1; ++dx) {
						for (int dy = -1; dy <= 1; ++dy) {
							for (int dz = -1; dz <= 1; ++dz) {
								// Skip the current voxel
								if (dx == 0 && dy == 0 && dz == 0)
									continue;

								const Eigen::Vector3i neighborPosition = Eigen::Vector3i(x, y, z) + Eigen::Vector3i(dx, dy, dz);
								if (neighborPosition.x() < 0 || neighborPosition.x() >= dimensions_.x() ||
									neighborPosition.y() < 0 || neighborPosition.y() >= dimensions_.y() ||
									neighborPosition.z() < 0 || neighborPosition.z() >= dimensions_.z())
								{
									isBoundaryVoxel = true;
									break;
								}

								const Voxel& neighbor = GetVoxel(neighborPosition);

								// If a neighboring voxel is empty, mark the current voxel as a boundary voxel
								if (neighbor.value == 0) {
									isBoundaryVoxel = true;
									break;
								}
							}
							if (isBoundaryVoxel)
								break;
						}
						if (isBoundaryVoxel)
							break;
					}
					if (isBoundaryVoxel)
						boundaryVoxels.push_back(Eigen::Vector3i(x, y, z));
				}
			}
		}
		return boundaryVoxels;
	}

    const Eigen::Vector3d& GetVoxelCenter(const Eigen::Vector3i& voxelPosition) {
        Eigen::Vector3i vp(voxelPosition.x(), voxelPosition.y(), -voxelPosition.z());
        return origin_ + voxelCenterOffset_ + (voxelSize_ * vp.cast<double>());
    }

	static VoxelGrid CreateFilledVoxelGrid(const Eigen::Vector3d& origin, const Eigen::Vector3i& dimensions, double voxelSize = 1.0, int fillValue = 1) {

		VoxelGrid voxelGrid(voxelSize, origin, dimensions);

		std::fill(voxelGrid.voxels_.begin(), voxelGrid.voxels_.end(), Voxel(fillValue));

		return voxelGrid;
	}
	static VoxelGrid GetZeroEnclosedVoxelGrid(VoxelGrid& voxelgrid) {
		VoxelGrid newVoxelGrid(voxelgrid.voxelSize_, voxelgrid.origin_ - Eigen::Vector3d(1, 1, 1) * voxelgrid.voxelSize_, voxelgrid.dimensions_ + Eigen::Vector3i(2, 2, 2));
		std::fill(newVoxelGrid.voxels_.begin(), newVoxelGrid.voxels_.end(), Voxel(0));
		const std::vector<Eigen::Vector3i> setVoxels = voxelgrid.GetSetVoxels();
		for (auto voxel : setVoxels) {
			newVoxelGrid.AddVoxel(voxel + Eigen::Vector3i(1, 1, 1), Voxel(1));
		}
		return newVoxelGrid;
	}

	const cv::Vec3b GetVoxelColor(const Eigen::Vector3i& voxelPosition) {
		// gets average color (Other option would be dominant color)
		std::vector<cv::Vec3b> colors;
		Eigen::Vector3d voxelWorldPos = GetVoxelCenter(voxelPosition);
		Voxel v = GetVoxel(voxelPosition);
		for (Camera c : v.cameras) {
			Eigen::Vector2i cameraPos = c.ProjectIntoCameraSpace(voxelWorldPos);
			colors.push_back(c.frame.at<cv::Vec3b>(cameraPos.y(), cameraPos.x()));
		}
		cv::Vec3i sumColor(0, 0, 0);
		for (const cv::Vec3b& color : colors) {
			sumColor += cv::Vec3i(color[0], color[1], color[2]);
		}

		int numColors = colors.size();
		if (numColors == 0)
			return cv::Vec3b(255, 255, 255);
    
		cv::Vec3b averageColor(
			static_cast<unsigned char>(sumColor[0] / numColors),
			static_cast<unsigned char>(sumColor[1] / numColors),
			static_cast<unsigned char>(sumColor[2] / numColors)
		);

		return averageColor;
	}
	const cv::Vec3b GetVoxelColor(const int& x, const int& y, const int& z)
	{
		return GetVoxelColor(Eigen::Vector3i(x, y, z));
	}

private:
	double voxelSize_;
	Eigen::Vector3d origin_;
	Eigen::Vector3d voxelCenterOffset_;
	Eigen::Vector3i dimensions_;
	std::vector<Voxel> voxels_;

	int GetIndex(const Eigen::Vector3i& position) const {
		int index = position.x() + dimensions_.x() * (position.y() + dimensions_.y() * position.z());
		return index;
	}
};

