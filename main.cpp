#include "voxel/VoxelGrid.h"
#include <iostream>
#include "voxel/VoxelGridExporter.h"

#define RunVoxelGridTest 1

const std::string voxeTestFilenameTarget = std::string("voxelGrid.off");

int main() {
	if (RunVoxelGridTest) {
		auto grid = VoxelGrid::CreateFilledVoxelGrid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3i(50, 50, 50), 1);
		std::cout << grid.GetVoxelCount() << std::endl;
		auto v = grid.GetVoxelCenter(Eigen::Vector3i(1, 0, 0));
		for (int i = 0; i < 50; i++) {
			for (int j = 30; j < 40; j++) {
				for (int k = 30; k < 40; k++) {
					grid.RemoveVoxel(Eigen::Vector3i(i, j, k));
				}
			}
		}
		VoxelGridExporter::ExportToOFF(voxeTestFilenameTarget, grid);
	}
	return 0;
}
