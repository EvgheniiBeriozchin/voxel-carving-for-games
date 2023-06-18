#pragma once
#include "VoxelGrid.h"

class Camera {
	// TODO: PixelGrid / image for marking while sweeping
public:
	const Eigen::Vector2i& ProjectIntoCameraSpace(Eigen::Vector3d worldPoint){
		// TODO: project into camera pixel space
	}
	bool IsMarked(Eigen::Vector2i& pixel) {
		// TODO: Check if pixel is unmarked
	}
	void MarkPixel(Eigen::Vector2i& pixel) {
		// TODO: Mark pixel
	}
};


class SpaceCarver {
public:
	enum SpaceCarvingDirection { XPos, XNeg, YPos, YNeg, ZPos, ZNeg };

	static void PlaneSweep(VoxelGrid& voxel_grid, std::vector<Camera> cameras, SpaceCarvingDirection direction) {
		Eigen::Vector3d planeNormal;
		int xStart, xEnd, yStart, yEnd, zStart, zEnd;
		xStart = yStart = zStart = 0;
		xEnd = voxel_grid.GetDimensions().x();
		yEnd = voxel_grid.GetDimensions().y();
		zEnd = voxel_grid.GetDimensions().z();
		switch (direction)
		{
		case SpaceCarvingDirection::XPos:
			planeNormal = Eigen::Vector3d(-1, 0, 0);
			xEnd = 0;// Every single sweep only 1 layer at a time
			for (int x = 0; x < voxel_grid.GetDimensions().x(); x++) {
				SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::XNeg:
			planeNormal = Eigen::Vector3d(1, 0, 0);
			xEnd = 0;
			for (int x = voxel_grid.GetDimensions().x() -1; x >= 0; x--) {
				SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::YPos:
			planeNormal = Eigen::Vector3d(0, -1, 0);
			yEnd = 0;
			for (int y = 0; y < voxel_grid.GetDimensions().y(); y++) {
				SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::YNeg:
			planeNormal = Eigen::Vector3d(0, 1, 0);
			yEnd = 0;
			for (int y = voxel_grid.GetDimensions().y() - 1; y >= 0; y--) {
				SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::ZPos:
			planeNormal = Eigen::Vector3d(0, 0 , -1);
			zEnd = 0;
			for (int z = 0; z < voxel_grid.GetDimensions().z(); z++) {
				SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::ZNeg:
			planeNormal = Eigen::Vector3d(0, 0, 1);
			zEnd = 0;
			for (int z = voxel_grid.GetDimensions().z() - 1; z >= 0; z--) {
				SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, zStart, zEnd);
			}
			break;
		default:
			break;
		}
	}

	static void MultiSweep(VoxelGrid& voxel_grid, std::vector<Camera> cameras) {
		//TODO: Step3 / Step4
		// Plane sweep in all directions
		PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::XPos);
		PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::XNeg);
		PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::YPos);
		PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::YNeg);
		PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::ZPos);
		PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::ZNeg);
		//TODO: Change Voxelgrid to use properties like cameras/ consistency-check-counts
	}
private:
	static void SinglePlaneSweep(VoxelGrid& voxel_grid, std::vector<Camera> cameras, const Eigen::Vector3d& planeNormal, int xStart, int xEnd, int yStart, int yEnd, int zStart, int zEnd) {
		// Peel a single layer
		for (int x = xStart; x <= xEnd; x++) {
			for (int y = yStart; y <= yEnd; y++) {
				for (int z = zStart; z <= zEnd; z++) {
					Eigen::Vector3i voxel_grid_pos = Eigen::Vector3i(x, y, z);
					if (!IsSurfaceVoxel(voxel_grid, voxel_grid_pos))// Skip voxels that are the interior of the object (no need to check them TODO: validate if thats correct)
						continue;
					Voxel& voxel = voxel_grid.GetVoxel(voxel_grid_pos);
					Eigen::Vector3d voxel_world_pos = voxel_grid.GetVoxelCenter(voxel_grid_pos);
					std::vector<Camera> cameras = GetCamerasForPlaneVoxel(voxel_world_pos, planeNormal, cameras);// Get cameras above plane
					std::vector<Eigen::Vector2i> pixelsPositions;
					std::vector<Camera> unmarkedPixelCameras;
					for (int i = 0; i < cameras.size(); i++) {// Select cameras and pixel positions where pixel is unmarked
						Eigen::Vector2i pixelPos = cameras[i].ProjectIntoCameraSpace(voxel_world_pos);
						if (!cameras[i].IsMarked(pixelPos)) {
							unmarkedPixelCameras.push_back(cameras[i]);
							pixelsPositions.push_back(pixelPos);
						}
					}
					if (CheckPhotoConsistency(voxel_world_pos, pixelsPositions, unmarkedPixelCameras)) { // Check Photo consistency
						for (int i = 0; i < unmarkedPixelCameras.size(); i++)
						{
							unmarkedPixelCameras[i].MarkPixel(pixelsPositions[i]);
						}
					}
					else {// carve voxel
						voxel_grid.RemoveVoxel(voxel_grid_pos);
					}
				}
			}
		}
	}
	static bool IsSurfaceVoxel(VoxelGrid& voxel_grid, Eigen::Vector3i voxel_pos) {
		auto dims = voxel_grid.GetDimensions();
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dz = -1; dz <= 1; ++dz) {
					if (dx == 0 && dy == 0 && dz == 0)
						continue;
					const Eigen::Vector3i neighborPosition = voxel_pos + Eigen::Vector3i(dx, dy, dz);
					if (neighborPosition.x() < 0 || neighborPosition.x() >= dims.x() ||
						neighborPosition.y() < 0 || neighborPosition.y() >= dims.y() ||
						neighborPosition.z() < 0 || neighborPosition.z() >= dims.z())
					{
						return true;	// voxel_grid boundary is always surface
					}
					const Voxel& neighbor = voxel_grid.GetVoxel(neighborPosition);
					if (neighbor.value == 0) {// a neighbor is not set => is on surface
						return true;
					}
				}
			}
		}
		return false;	// fully enclosed in other set voxels
	}

	static std::vector<Camera> GetCamerasForPlaneVoxel(const Eigen::Vector3d& voxel_world_pos, const Eigen::Vector3d& planeNormal, const std::vector<Camera>& cameras) {
		std::vector<Camera> cams;
		for each (Camera camera in cameras)
		{
			if (IsCameraAbovePlane(camera, voxel_world_pos, planeNormal))
				cams.push_back(camera);
		}
		return cams;
	}
	static bool IsCameraAbovePlane(const Camera& camera, const Eigen::Vector3d& planePoint, const Eigen::Vector3d& planeNormal) {
		// for better result use clipping in pyramidal beam instead of plane
		// TODO:
		// double distance = (camera.point - planePoint).dot(planeNormal);
		// return distance > 0.0;
		return false;
	}
	static bool CheckPhotoConsistency(const Eigen::Vector3d& voxel_world_pos, const std::vector<Eigen::Vector2i>& pixelsPositions, const std::vector<Camera>& PixelCameras) {
		//TODO:
	}

};