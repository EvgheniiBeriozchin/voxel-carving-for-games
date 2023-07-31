#pragma once

#ifndef MARCHING_CUBES2_H
#define MARCHING_CUBES2_H

#include "voxel/SimpleMesh.h"
#include "Volume.h"

//****************************************************************************************************************
/*
Linearly interpolate the position where an isosurface cuts
an edge between two vertices, each with their own scalar value
*/
Eigen::Vector3d VertexInterp2(double isolevel, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double valp1, double valp2)
{
	// TODO: implement the linear interpolant
	// Assume that the function value at 'p1' is 'valp1' and the function value at 'p2' is 'valp2'.
	// Further assume that the function is linear between 'p1' and 'p2'. Compute and return the
	// point 'p' on the line from 'p1' to 'p2' where the function takes on the value 'isolevel'
	//
	//       f(p2) = valp2
	//       x
	//      /
	//     x f(p) = isolevel
	//    /
	//   /
	//  /
	// x
	// f(p1) = valp1
	
	return p1 + (p2 - p1) * (1 - abs(valp2 - isolevel) / abs(valp2 - valp1));

	// return (p1+p2)/2;
}

/*
Given a grid cell and an isolevel, calculate the triangular
facets required to represent the isosurface through the cell.
Return the number of triangular facets, the array "triangles"
will be loaded up with the vertices at most 5 triangular facets.
0 will be returned if the grid cell is either totally above
or totally below the isolevel.
*/
int Polygonise2(MC_Gridcell grid, double isolevel, MC_Triangle* triangles) {

	int ntriang;
	int cubeindex;
	Eigen::Vector3d vertlist[12];

	cubeindex = 0;
	if (grid.val[0] < isolevel) cubeindex |= 1;
	if (grid.val[1] < isolevel) cubeindex |= 2;
	if (grid.val[2] < isolevel) cubeindex |= 4;
	if (grid.val[3] < isolevel) cubeindex |= 8;
	if (grid.val[4] < isolevel) cubeindex |= 16;
	if (grid.val[5] < isolevel) cubeindex |= 32;
	if (grid.val[6] < isolevel) cubeindex |= 64;
	if (grid.val[7] < isolevel) cubeindex |= 128;

	/* Cube is entirely in/out of the surface */
	if (edgeTable[cubeindex] == 0)
		return 0;

	/* Find the vertices where the surface intersects the cube */
	if (edgeTable[cubeindex] & 1)
		vertlist[0] = VertexInterp2(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
	if (edgeTable[cubeindex] & 2)
		vertlist[1] = VertexInterp2(isolevel, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
	if (edgeTable[cubeindex] & 4)
		vertlist[2] = VertexInterp2(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
	if (edgeTable[cubeindex] & 8)
		vertlist[3] = VertexInterp2(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
	if (edgeTable[cubeindex] & 16)
		vertlist[4] = VertexInterp2(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
	if (edgeTable[cubeindex] & 32)
		vertlist[5] = VertexInterp2(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
	if (edgeTable[cubeindex] & 64)
		vertlist[6] = VertexInterp2(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
	if (edgeTable[cubeindex] & 128)
		vertlist[7] = VertexInterp2(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
	if (edgeTable[cubeindex] & 256)
		vertlist[8] = VertexInterp2(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
	if (edgeTable[cubeindex] & 512)
		vertlist[9] = VertexInterp2(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
	if (edgeTable[cubeindex] & 1024)
		vertlist[10] = VertexInterp2(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
	if (edgeTable[cubeindex] & 2048)
		vertlist[11] = VertexInterp2(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

	/* Create the triangle */
	ntriang = 0;
	for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
		triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i]];
		triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i + 1]];
		triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i + 2]];
		ntriang++;
	}

	return ntriang;
}


bool ProcessVolumeCell2(Volume* vol, int x, int y, int z, double iso, SimpleMesh* mesh)
{
	MC_Gridcell cell;

	Eigen::Vector3d tmp;

	// cell corners
	tmp = vol->pos(x + 1, y, z);
	cell.p[0] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x, y, z);
	cell.p[1] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x, y + 1, z);
	cell.p[2] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x + 1, y + 1, z);
	cell.p[3] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x + 1, y, z + 1);
	cell.p[4] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x, y, z + 1);
	cell.p[5] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x, y + 1, z + 1);
	cell.p[6] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);
	tmp = vol->pos(x + 1, y + 1, z + 1);
	cell.p[7] = Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);

	// cell corner values
	cell.val[0] = (double)vol->get(x + 1, y, z);
	cell.val[1] = (double)vol->get(x, y, z);
	cell.val[2] = (double)vol->get(x, y + 1, z);
	cell.val[3] = (double)vol->get(x + 1, y + 1, z);
	cell.val[4] = (double)vol->get(x + 1, y, z + 1);
	cell.val[5] = (double)vol->get(x, y, z + 1);
	cell.val[6] = (double)vol->get(x, y + 1, z + 1);
	cell.val[7] = (double)vol->get(x + 1, y + 1, z + 1);

	MC_Triangle tris[6];
	int numTris = Polygonise2(cell, iso, tris);

	if (numTris == 0)
		return false;

	for (int i1 = 0; i1 < numTris; i1++)
	{
		Vertex v0((float)tris[i1].p[0][0], (float)tris[i1].p[0][1], (float)tris[i1].p[0][2]);
		Vertex v1((float)tris[i1].p[1][0], (float)tris[i1].p[1][1], (float)tris[i1].p[1][2]);
		Vertex v2((float)tris[i1].p[2][0], (float)tris[i1].p[2][1], (float)tris[i1].p[2][2]);

		unsigned int vhandle[3];
		vhandle[0] = mesh->AddVertex(v0);
		vhandle[1] = mesh->AddVertex(v1);
		vhandle[2] = mesh->AddVertex(v2);

		mesh->AddFace(vhandle[0], vhandle[1], vhandle[2]);
	}

	return true;
}

#endif // MARCHING_CUBES2_H
