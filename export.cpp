#include <stdio.h>
#include <stdlib.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
using namespace glm;

using namespace std;

#include <iostream>
#include <string>
#include <Eigen/Core>

#include <TutteEmbedding.h>

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

#include "export.h"
using namespace MeshExport;

GLuint LoadShaders(const char* vertex_file_path, const char* fragment_file_path) {

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	string VertexShaderCode;
	ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open()) {
		stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}
	else {
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	string FragmentShaderCode;
	ifstream FragmentShaderStream(fragment_file_path, ios::in);
	if (FragmentShaderStream.is_open()) {
		stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;


	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const* VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}



	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const* FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}



	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}


	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

void MeshExport::RenderTexture(std::string textureName, Eigen::MatrixXd& U, Eigen::MatrixXi& F, Eigen::MatrixXd& Colors)
{
	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1920, 1920, "Texture Render", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders("TransformVertexShader.vertexshader", "ColorFragmentShader.fragmentshader");

	vector<GLfloat> g_vertex_buffer_data;
	vector<GLfloat> g_color_buffer_data;

	for (int i = 0; i < F.rows(); i++) {
		//Get the three vertices of the face
		for (int j = 0; j < 3; j++)
		{
			int vertexIndex = F(i, j);
			//Input points are UV mapped [0,1]x[0,1] so we need to scale them to [-1,1]x[-1,1]
			float u = U(vertexIndex, 0) * 2 - 1;
			float v = U(vertexIndex, 1) * 2 - 1;
			//uvs origin is top left, opengl is bottom left
			v = -v;
			g_vertex_buffer_data.push_back(u);
			g_vertex_buffer_data.push_back(v);
			g_vertex_buffer_data.push_back(0.0f);

			g_color_buffer_data.push_back(Colors(vertexIndex, 0));
			g_color_buffer_data.push_back(Colors(vertexIndex, 1));
			g_color_buffer_data.push_back(Colors(vertexIndex, 2));
		}
	}

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, g_vertex_buffer_data.size() * sizeof(GLfloat), &g_vertex_buffer_data[0], GL_STATIC_DRAW);

	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, g_color_buffer_data.size() * sizeof(GLfloat), &g_color_buffer_data[0], GL_STATIC_DRAW);


	glfwHideWindow(window);

	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT);

	// Use our shader
	glUseProgram(programID);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	// 2nd attribute buffer : colors
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// Draw the triangle !
	glDrawArrays(GL_TRIANGLES, 0, g_vertex_buffer_data.size()); // 3 indices starting at 0 -> 1 triangle

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();

	std::cout << "Writing texture to file: " << textureName << ".png" << std::endl;

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight); /* Get size, store into specified variables  */

	/* Prepare the targa header */
	memcpy(tga.head, tgahead, 12);
	tga.dx = screenWidth;
	tga.dy = screenHeight;
	tga.head2 = 0x2018;

	/* Store pixels into tga.pic */
	glReadPixels(0, 0, screenWidth, screenHeight, GL_RGB, GL_UNSIGNED_BYTE, tga.pic[0]);

	std::string filename = textureName + ".png";

	stbi_write_png(filename.c_str(), screenWidth, screenHeight, 3, tga.pic[0], screenWidth * 3);

	std::cout << "Texture written to file" << std::endl;

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

}

void MeshExport::WriteObj(string objName, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N, Eigen::MatrixXd& Col)
{
	std::cout << "Writing OBJ file " << objName << ".obj" << std::endl;
	ofstream myfile;
	string filename = objName + ".obj";


	size_t lastindex = filename.find_last_of("/");
	std::string path = filename.substr(0, lastindex);
	std::string fileName = filename.substr(lastindex + 1);
	std::string fileNameNoEnding = fileName.substr(0, fileName.find_last_of("."));

	//Create .mtl file reference
	std::string mtlFileName = objName + ".mtl";

	myfile.open(filename);

	//Add material file reference
	myfile << "mtllib " << fileNameNoEnding << ".mtl" << std::endl;

	//Create object
	myfile << "o " << fileNameNoEnding << std::endl;

	//Add material to object. we only create one default material
	myfile << "usemtl " << "Default" << std::endl;


	//Write vertices. Add vertex color as backup for those applications that support it
	for (int i = 0; i < V.rows(); i++)
	{
		//myfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
		myfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " " << Col(i, 0) << " " << Col(i, 1) << " " << Col(i, 2) << std::endl;
	}

	//Write texture coordinates
	for (int i = 0; i < U.rows(); i++)
	{
		myfile << "vt " << U(i, 0) << " " <<  U(i, 1) << std::endl;
	}

	//Write normals
	for (int i = 0; i < N.rows(); i++)
	{
		myfile << "vn " << N(i, 0) << " " << N(i, 1) << " " << N(i, 2) << std::endl;
	}

	//Write faces, include vertex normals and texture coordinates
	for (int i = 0; i < F.rows(); i++)
	{
		myfile << "f " << F(i, 0) + 1 << "/" << F(i, 0) + 1 << "/" << F(i, 0) + 1 << " " << F(i, 1) + 1 << "/" << F(i, 1) + 1 << "/" << F(i, 1) + 1 << " " << F(i, 2) + 1 << "/" << F(i, 2) + 1 << "/" << F(i, 2) + 1 << std::endl;
	}

	myfile.close();



	std::cout << "Object file written" << std::endl;
	std::cout << "Writing MTL file " << mtlFileName << std::endl;

	//Create .mtl file
	myfile.open(mtlFileName);
	myfile << "newmtl Default" << std::endl;
	myfile << "Ka 1.000000 1.000000 1.000000" << std::endl;
	myfile << "Kd 1.000000 1.000000 1.000000" << std::endl;
	myfile << "Ks 0.000000 0.000000 0.000000" << std::endl;
	myfile << "Ns 0.000000" << std::endl;
	myfile << "d 1.000000" << std::endl;
	myfile << "illum 0" << std::endl;
	myfile << "map_Kd " << fileNameNoEnding << ".png" << std::endl;
	myfile.close();

	std::cout << "MTL file written" << std::endl;

}
