#pragma once
#include <GL/glew.h>
#include <Eigen/Core>


namespace MeshExport {

	typedef struct
	{
		unsigned char head[12];
		unsigned short dx /* Width */, dy /* Height */, head2;
		unsigned char pic[768 * 1024 * 10][3];
	} typetga;

	static typetga tga;

	unsigned char const tgahead[12] = { 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

	void RenderTexture(std::string textureName, Eigen::MatrixXd& U, Eigen::MatrixXi& F, Eigen::MatrixXd& Colors);

	void WriteObj(std::string objName, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N, Eigen::MatrixXd& Col);
}