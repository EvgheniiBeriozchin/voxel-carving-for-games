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
    static void ExportToPLY(const std::string& filename, const std::vector<std::vector<Eigen::Vector3d>> points, const std::vector<Eigen::Vector3d> colors) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write the OFF file header
       

        int pointsSize = 0;
        for (int i = 0; i < points.size(); i++) {
            pointsSize += points[i].size();
        }

        file << "ply" << std::endl;
        file << "format ascii 1.0" << std::endl;
        file << "element vertex " << pointsSize << std::endl;
        file << "property float x" << std::endl;
        file << "property float y" << std::endl;
        file << "property float z" << std::endl;
        file << "property uchar red" << std::endl;
        file << "property uchar green" << std::endl;
        file << "property uchar blue" << std::endl;
        file << "end_header" << std::endl;

        // Write the voxel grid vertices
        for (int i = 0; i < points.size(); i++) {
            for (int z = 0; z < points[i].size(); z++) {
                file << points[i][z].x() << " " << points[i][z].y() << " " << points[i][z].z() << " " << colors[i].x() << " " << colors[i].y() << " " << colors[i].z() << std::endl;
            }
        }



        file.close();
    }

private:

};

