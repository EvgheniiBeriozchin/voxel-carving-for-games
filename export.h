
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>

#include <Eigen/Core>
#include <TutteEmbedding.h>

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION


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