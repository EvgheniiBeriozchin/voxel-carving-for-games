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



struct IglInputFormat {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd Colors;
};

class SimpleMesh
{
public:

	void Clear()
	{
		m_vertices.clear();
		m_triangles.clear();
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

	IglInputFormat GetIglInputFormat()
	{
		IglInputFormat iglInputFormat;
		iglInputFormat.V.resize(m_vertices.size(), 3);
		iglInputFormat.F.resize(m_triangles.size(), 3);
		iglInputFormat.Colors.resize(m_colors.size(), 3);
		for (unsigned int i = 0; i < m_vertices.size(); i++)
		{
			iglInputFormat.V.row(i) = m_vertices[i];
		}
		for (unsigned int i = 0; i < m_triangles.size(); i++)
		{
			iglInputFormat.F.row(i) = Eigen::Vector3i(m_triangles[i].idx0, m_triangles[i].idx1, m_triangles[i].idx2);
		}

		for (unsigned int i = 0; i < m_colors.size(); i++)
		{
			iglInputFormat.Colors.row(i) = Eigen::Vector3d(m_colors[i][0], m_colors[i][1], m_colors[i][2]);
		}

		return iglInputFormat;
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
