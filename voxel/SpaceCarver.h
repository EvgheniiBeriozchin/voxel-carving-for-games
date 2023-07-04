#pragma once
#include "VoxelGrid.h"
#include "../Camera.h"



class SpaceCarver {
public:
	enum SpaceCarvingDirection { XPos, XNeg, YPos, YNeg, ZPos, ZNeg };

	static bool PlaneSweep(VoxelGrid& voxel_grid, std::vector<Camera>& cameras, SpaceCarvingDirection direction) {
		Eigen::Vector3d planeNormal;
		bool removed = false;
		int xStart, xEnd, yStart, yEnd, zStart, zEnd;
		xStart = yStart = zStart = 0;
		xEnd = voxel_grid.GetDimensions().x() -1;
		yEnd = voxel_grid.GetDimensions().y() -1;
		zEnd = voxel_grid.GetDimensions().z() -1;
		switch (direction)
		{
		case SpaceCarvingDirection::XPos:
			planeNormal = Eigen::Vector3d(-1, 0, 0);
			xEnd = 0;// Every single sweep only 1 layer at a time
			for (int x = 0; x < voxel_grid.GetDimensions().x(); x++) {
				removed |= SinglePlaneSweep(voxel_grid, cameras, planeNormal, x, x, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::XNeg:
			planeNormal = Eigen::Vector3d(1, 0, 0);
			xEnd = 0;
			for (int x = voxel_grid.GetDimensions().x() -1; x >= 0; x--) {
				removed |= SinglePlaneSweep(voxel_grid, cameras, planeNormal, x, x, yStart, yEnd, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::YPos:
			planeNormal = Eigen::Vector3d(0, -1, 0);
			yEnd = 0;
			for (int y = 0; y < voxel_grid.GetDimensions().y(); y++) {
				removed |= SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, y, y, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::YNeg:
			planeNormal = Eigen::Vector3d(0, 1, 0);
			yEnd = 0;
			for (int y = voxel_grid.GetDimensions().y() - 1; y >= 0; y--) {
				removed |= SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, y, y, zStart, zEnd);
			}
			break;
		case SpaceCarvingDirection::ZPos:
			planeNormal = Eigen::Vector3d(0, 0 , -1);
			zEnd = 0;
			for (int z = 0; z < voxel_grid.GetDimensions().z(); z++) {
				removed |= SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, z, z);
			}
			break;
		case SpaceCarvingDirection::ZNeg:
			planeNormal = Eigen::Vector3d(0, 0, 1);
			zEnd = 0;
			for (int z = voxel_grid.GetDimensions().z() - 1; z >= 0; z--) {
				removed |= SinglePlaneSweep(voxel_grid, cameras, planeNormal, xStart, xEnd, yStart, yEnd, z, z);
			}
			break;
		default:
			break;
		}
		return removed;
	}

	static void MultiSweep(VoxelGrid& voxel_grid, std::vector<Camera>& cameras) {
		//TODO: Step3 / Step4
		// Plane sweep in all directions
		auto c = cameras;
		bool terminate = false;
		while (!terminate)// continue unit no change
		{
			bool change = false;
			// Step 2
			change |= PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::XPos);
			change |= PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::XNeg);
			change |= PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::YPos);
			change |= PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::YNeg);
			change |= PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::ZPos);
			change |= PlaneSweep(voxel_grid, cameras, SpaceCarvingDirection::ZNeg);
			// Step 3
			change |= MultiSweepConsistency(voxel_grid, cameras);
			terminate = !change;
		}
	}
private:
	static bool SinglePlaneSweep(VoxelGrid& voxel_grid, std::vector<Camera>& cameras, const Eigen::Vector3d& planeNormal, int xStart, int xEnd, int yStart, int yEnd, int zStart, int zEnd) {
		bool removed = false;
		// Peel a single layer
		for (int x = xStart; x <= xEnd; x++) {
			for (int y = yStart; y <= yEnd; y++) {
				for (int z = zStart; z <= zEnd; z++) {
					Eigen::Vector3i voxel_grid_pos = Eigen::Vector3i(x, y, z);
					if (!IsSurfaceVoxel(voxel_grid, voxel_grid_pos))// Skip voxels that are the interior of the object (no need to check them TODO: validate if thats correct)
						continue;
					Voxel& voxel = voxel_grid.GetVoxel(voxel_grid_pos);
					Eigen::Vector3d voxel_world_pos = voxel_grid.GetVoxelCenter(voxel_grid_pos);
					std::vector<Camera> voxelCameras = GetCamerasForPlaneVoxel(voxel_world_pos, planeNormal, cameras);// Get cameras above plane
					std::vector<Eigen::Vector2i> pixelsPositions;
					std::vector<Camera> unmarkedPixelCameras;
					if (voxelCameras.size() == 0)
						continue;
					for (int i = 0; i < voxelCameras.size(); i++) {// Select cameras and pixel positions where pixel is unmarked
						Eigen::Vector2i pixelPos = voxelCameras[i].ProjectIntoCameraSpace(voxel_world_pos);
						if (pixelPos.x() < 0 || pixelPos.y() < 0 || pixelPos.x() >= voxelCameras[i].frame.rows || pixelPos.y() >= voxelCameras[i].frame.cols)
							continue;

						if (!voxelCameras[i].IsMarked(pixelPos)) {
							unmarkedPixelCameras.push_back(cameras[i]);
							pixelsPositions.push_back(pixelPos);
						}
					}
					if (CheckPhotoConsistency(voxel_world_pos, pixelsPositions, unmarkedPixelCameras)) { // Check Photo consistency
						for (int i = 0; i < unmarkedPixelCameras.size(); i++)
						{
							unmarkedPixelCameras[i].MarkPixel(pixelsPositions[i]);
							voxel.cameras.push_back(unmarkedPixelCameras[i]);
						}
					}
					else {// carve voxel
						voxel_grid.RemoveVoxel(voxel_grid_pos);
						removed = true;
					}
				}
			}
		}
		return removed;
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

	static std::vector<Camera> GetCamerasForPlaneVoxel(const Eigen::Vector3d& voxel_world_pos, const Eigen::Vector3d& planeNormal, const std::vector<Camera> cameras) {
		std::vector<Camera> cams;
		for each (Camera camera in cameras)
		{
			auto pixelPos = camera.ProjectIntoCameraSpace(voxel_world_pos);
			if (pixelPos.x() >= camera.frame.rows || pixelPos.y() >= camera.frame.cols || pixelPos.x() < 0 || pixelPos.y() < 0)
				continue;
			if (IsCameraAbovePlane(camera, voxel_world_pos, planeNormal))
				cams.push_back(camera);
		}
		return cams;
	}
	static bool IsCameraAbovePlane(const Camera& camera, const Eigen::Vector3d& planePoint, const Eigen::Vector3d& planeNormal) {
		// for better result use clipping in pyramidal beam instead of plane
		// TODO:
		//Eigen::Vector3d cp = camera.pose.position.cast<double>();
		//Eigen::Vector3d cr = camera.pose.rotation.cast<double>();

		//bool distance = (planePoint - cp).normalized().dot(planeNormal) > 0;
		//bool direction = cr.dot(planeNormal) > 0;
		//// return distance > 0.0;
		//return distance && direction;
		return true;
	}
	static bool CheckPhotoConsistency(const Eigen::Vector3d& voxel_world_pos, const std::vector<Eigen::Vector2i>& pixelsPositions, const std::vector<Camera> PixelCameras) {
		bool consistent = true;
		for (int i = 0; i < pixelsPositions.size(); i++) {
			Camera c = PixelCameras[i];
			uchar col = c.grayScaleFrame.at<uchar>(pixelsPositions[i].x(), pixelsPositions[i].y());
			if (col >= 10) {
				consistent &= false;
			}
		}
		return consistent;
	}

	static bool MultiSweepConsistency(VoxelGrid& voxel_grid, const std::vector<Camera> cameras) {
		std::vector<Eigen::Vector3i> setVoxelPositions = voxel_grid.GetSetVoxels();
		bool removed = false;
		for (int i = 0; i < setVoxelPositions.size(); i++) {
			std::vector<Eigen::Vector2i> pixelsPositions;
			Voxel& voxel = voxel_grid.GetVoxel(setVoxelPositions[i]);
			Eigen::Vector3d voxel_world_pos = voxel_grid.GetVoxelCenter(setVoxelPositions[i]);
			for (int c = 0; c < voxel.cameras.size(); c++) {
				Eigen::Vector2i pixelPos = voxel.cameras[c].ProjectIntoCameraSpace(voxel_world_pos);
				pixelsPositions.push_back(pixelPos);
			}
			if (!CheckPhotoConsistency(voxel_world_pos, pixelsPositions, voxel.cameras)) {// Step 3b
				voxel_grid.RemoveVoxel(setVoxelPositions[i]);
				removed = true;
			}
		}
		return removed;
	}
};