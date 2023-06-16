#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "VoxelGrid.h"

class VoxelGridExporter {
public:
    static void ExportToOBJ(const std::string& filePath, VoxelGrid& voxelGrid) {
        std::ofstream outputFile(filePath);
        if (!outputFile) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            return;
        }

        ExportPointCloud(outputFile, voxelGrid);

        outputFile.close();
        std::cout << "VoxelGrid exported as point cloud to OBJ file: " << filePath << std::endl;
    }

    static void ExportToOFF(const std::string& filename, VoxelGrid& voxelGrid) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write the OFF file header
        file << "OFF" << std::endl;
        std::vector<Eigen::Vector3d> voxels = voxelGrid.GetSetVoxelCenterPoints();
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

    static void ExportPointCloud(std::ofstream& outputFile, VoxelGrid& voxelGrid) {
        const Eigen::Vector3i& gridSize = voxelGrid.GetDimensions();
        double voxelSize = voxelGrid.GetVoxelSize();

        for (int z = 0; z < gridSize.z(); ++z) {
            for (int y = 0; y < gridSize.y(); ++y) {
                for (int x = 0; x < gridSize.x(); ++x) {
                    Eigen::Vector3i voxelPosition(x, y, z);
                    const Eigen::Vector3d& voxelCenter = voxelGrid.GetVoxelCenter(voxelPosition);

                    outputFile << "v " << voxelCenter.x() << " " << voxelCenter.y() << " " << voxelCenter.z() << std::endl;
                }
            }
        }
    }
};

