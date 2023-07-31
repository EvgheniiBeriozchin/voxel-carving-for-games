#pragma once

#ifndef SIMPLE_MESH_H
#define SIMPLE_MESH_H

#include <iostream>
#include <fstream>

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

typedef Eigen::Vector3d Vertex;
typedef Eigen::Vector3i Color;


struct Triangle
{
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;
	Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
		idx0(_idx0), idx1(_idx1), idx2(_idx2)
	{}
};

class SimpleMesh
{
public:

	void Clear()
	{
		m_vertices.clear();
		m_triangles.clear();
		m_colors.clear();
	}

	unsigned int AddVertex(Vertex& vertex)
	{
		unsigned int vId = (unsigned int)m_vertices.size();
		m_vertices.push_back(vertex);
		return vId;
	}

	unsigned int AddColor(cv::Vec3b& color) {
		unsigned int cId = (unsigned int)m_colors.size();
		m_colors.push_back(color);
		return cId;
	}

	unsigned int AddFace(unsigned int idx0, unsigned int idx1, unsigned int idx2)
	{
		unsigned int fId = (unsigned int)m_triangles.size();
		Triangle triangle(idx0, idx1, idx2);
		m_triangles.push_back(triangle);
		return fId;
	}

	std::vector<Vertex>& GetVertices()
	{
		return m_vertices;
	}

	std::vector<Triangle>& GetTriangles()
	{
		return m_triangles;
	}

	std::vector<cv::Vec3b>& GetColors()
	{
		return m_colors;
	}


	void DeduplicateVertices() {
		//Create map from position to vertice ids with that position
		std::map<std::vector<double>, std::vector<unsigned int>> positionToVertices;
		for (unsigned int i = 0; i < m_vertices.size(); i++) {
			//Position p(m_vertices[i][0], m_vertices[i][1], m_vertices[i][2]);
			std::vector<double> p = { m_vertices[i][0], m_vertices[i][1], m_vertices[i][2] };
			positionToVertices[p].push_back(i);
		}

		//Create new vertices and update triangle indices
		std::vector<Vertex> newVertices;
		std::vector<Triangle> newTriangles;
		std::vector<cv::Vec3b> newColors;


		std::map<int, int> oldToNewVertexIndex;

		for (std::map<std::vector<double>, std::vector<unsigned int>>::iterator it = positionToVertices.begin(); it != positionToVertices.end(); ++it) {
			auto position = it->first;
			auto vertices = it->second;

			//Create new vertex
			Vertex newVertex(position[0], position[1], position[2]);
			newVertices.push_back(newVertex);

			//Create new color
			Color newColor(m_colors[vertices[0]][0], m_colors[vertices[0]][1], m_colors[vertices[0]][2]);

			//Update old to new vertex index
			for (unsigned int i = 0; i < vertices.size(); i++) {
				oldToNewVertexIndex[vertices[i]] = newVertices.size() - 1;
			}
		}

		//Update triangle indices
		for (unsigned int i = 0; i < m_triangles.size(); i++) {
			Triangle triangle = m_triangles[i];
			int idx0 = triangle.idx0;
			int idx1 = triangle.idx1;
			int idx2 = triangle.idx2;

			if (idx0 == idx1 || idx0 == idx2 || idx1 == idx2) {
				std::cout << "Old Triangle with same indices" << std::endl;
				continue;
			}

			//Get new indices, make sure we go in the same order as before
			int newIdx0 = oldToNewVertexIndex[idx0];
			int newIdx1 = oldToNewVertexIndex[idx1];
			int newIdx2 = oldToNewVertexIndex[idx2];


			if (newIdx0 > newVertices.size() || newIdx1 > newVertices.size() || newIdx2 > newVertices.size()) {
				std::cout << "New Triangle with invalid indices" << std::endl;
			}
				


			if( newIdx0 == newIdx1 || newIdx0 == newIdx2 || newIdx1 == newIdx2)
				continue;

			//Create new triangle
			Triangle newTriangle(newIdx0, newIdx1, newIdx2);
			newTriangles.push_back(newTriangle);
		}

		/*
		//Update colors
		for (unsigned int i = 0; i < newVertices.size(); i++)
		{
			Position p(newVertices[i][0], newVertices[i][1], newVertices[i][2]);
			std::vector<unsigned int> vertices = positionToVertices[p];
			int idx = vertices[0];
			newColors.push_back(m_colors[idx]);
		}
		*/

		std::cout << "Deduplicated mesh from " << m_vertices.size() << " to " << newVertices.size() << " vertices" << std::endl;
		std::cout << "Deduplicated mesh from " << m_triangles.size() << " to " << newTriangles.size() << " triangles" << std::endl;

		//Update mesh
		m_vertices = newVertices;
		m_triangles = newTriangles;
		m_colors = newColors;
	}

	void GetMeshData(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &Colors)
	{
		V.resize(m_vertices.size(), 3);
		F.resize(m_triangles.size(), 3);
		Colors.resize(m_colors.size(), 3);
		for (unsigned int i = 0; i < m_vertices.size(); i++)
		{
			V.row(i) = m_vertices[i];
		}
		for (unsigned int i = 0; i < m_triangles.size(); i++)
		{
			F.row(i) = Eigen::Vector3i(m_triangles[i].idx0, m_triangles[i].idx1, m_triangles[i].idx2);
		}

		for (unsigned int i = 0; i < m_colors.size(); i++)
		{
			Colors.row(i) = Eigen::Vector3i(m_colors[i][2], m_colors[i][1], m_colors[i][0]);
		}
	}


	bool WriteMesh(const std::string& filename)
	{
		// Write off file
		std::ofstream outFile(filename);
		if (!outFile.is_open()) return false;

		// write header
		outFile << "OFF" << std::endl;
		outFile << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

		// save vertices
		for (unsigned int i = 0; i < m_vertices.size(); i++)
		{
			outFile << m_vertices[i].x() << " " << m_vertices[i].y() << " " << m_vertices[i].z() << std::endl;
		}

		// save faces
		for (unsigned int i = 0; i < m_triangles.size(); i++)
		{
			outFile << "3 " << m_triangles[i].idx0 << " " << m_triangles[i].idx1 << " " << m_triangles[i].idx2 << std::endl;
		}

		// close file
		outFile.close();

		return true;
	}

private:
	std::vector<Vertex> m_vertices;
	std::vector<Triangle> m_triangles;
	std::vector<cv::Vec3b> m_colors;
};

#endif // SIMPLE_MESH_H
