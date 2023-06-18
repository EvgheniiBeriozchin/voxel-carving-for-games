#pragma once

#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct Voxel {
	int value;
    Voxel() :value(0){}
	Voxel(int val) : value(val) {}
};

class VoxelGrid{
public:
	VoxelGrid(double voxelSize, const Eigen::Vector3d& origin) : VoxelGrid(voxelSize, origin, Eigen::Vector3i().Zero()) {
	}
    VoxelGrid(double voxelSize, const Eigen::Vector3d& origin, const Eigen::Vector3i& dimensions) :
        voxelSize_(voxelSize), 
        origin_(origin), 
        voxelCenterOffset_(Eigen::Vector3d(voxelSize / 2.0, voxelSize / 2.0, voxelSize / 2.0)),
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
        return origin_ + voxelCenterOffset_ + (voxelSize_ * voxelPosition.cast<double>());
    }

    static VoxelGrid CreateFilledVoxelGrid(const Eigen::Vector3d& origin, const Eigen::Vector3i& dimensions, double voxelSize = 1.0, int fillValue = 1) {

        VoxelGrid voxelGrid(voxelSize, origin, dimensions);

        std::fill(voxelGrid.voxels_.begin(), voxelGrid.voxels_.end(), Voxel(fillValue));

        return voxelGrid;
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

