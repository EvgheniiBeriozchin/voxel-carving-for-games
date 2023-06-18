#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "VoxelGrid.h"

class VoxelGridExporter {
public:
    static void ExportToOFF(const std::string& filename, VoxelGrid& voxelGrid) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write the OFF file header
        file << "OFF" << std::endl;
        std::vector<Eigen::Vector3i> voxelpositions = voxelGrid.GetBoundaryVoxels();
        std::vector<Eigen::Vector3d> voxels = voxelGrid.GetVoxelCenterPoints(voxelpositions);
        file << voxels.size() << " 0 0" << std::endl;

        // Write the voxel grid vertices
        for (const Eigen::Vector3d& position : voxels) {
            file << position.x() << " " << position.y() << " " << position.z() << std::endl;
        }

        // Write the voxel grid faces
        int index = 0;
        for (const Eigen::Vector3d& position : voxels) {
            file << "1 " << index << std::endl;
            index++;
        }
       

        file.close();
    }

private:

};

