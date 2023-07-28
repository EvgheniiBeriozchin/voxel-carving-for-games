// Include standard headers
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

#include <iostream>
#include <opencv2/opencv.hpp>

#include "voxel/VoxelGrid.h"
#include "voxel/VoxelGridExporter.h"
#include "voxel/SpaceCarver.h"
#include "Camera.h"
#include "utils/utils.h"
#include "TutteEmbedding.h"

#include <string>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/writePLY.h>
#include <igl/edges.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RUN_CAMERA_CALIBRATION 0
#define RUN_POSE_ESTIMATION_TEST 0
#define RUN_VOXEL_GRID_TEST 0
#define RUN_VOXEL_CARVING 0
#define RUN_UV_TEST 1


const int NUM_PROCESSED_FRAMES = 1000;
const std::string CALIBRATION_VIDEO_NAME = "../Box_NaturalLight.mp4";
const std::string RECONSTRUCTION_VIDEO_NAME = "../PepperMill_NaturalLight.mp4";
const std::string voxeTestFilenameTarget = std::string("voxelGrid.off");
const std::string uvTestingInput = "../bunny.off";

typedef struct
{
	unsigned char head[12];
	unsigned short dx /* Width */, dy /* Height */, head2;
	unsigned char pic[768 * 1024 * 10][3];
} typetga;
typetga tga;

unsigned char tgahead[12] = { 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };


GLuint LoadShaders(const char* vertex_file_path, const char* fragment_file_path) {

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
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
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::stringstream sstr;
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
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
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
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
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
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}


	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

void Barycentric(Eigen::Vector3d& p, Eigen::Vector3d& a, Eigen::Vector3d& b, Eigen::Vector3d& c, float& u, float& v, float& w) {
	Eigen::Vector3d v0 = b - a, v1 = c - a, v2 = p - a;
	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = d00 * d11 - d01 * d01;
	v = (d11 * d20 - d01 * d21) / denom;
	w = (d00 * d21 - d01 * d20) / denom;
	u = 1.0f - v - w;
}

void writeSvgFile(std::string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& VertexColors) {
	std::cout << "Writing svg file" << std::endl;

	std::ofstream myfile;
	myfile.open(filename);

	myfile << "<svg xmlns = \"http://www.w3.org/2000/svg\" version = \"1.1\">" << std::endl;


	for (unsigned int k = 0; k < F.rows(); k++) {
		Eigen::Vector2d p1 = U.row(F(k, 0));
		Eigen::Vector2d p2 = U.row(F(k, 1));
		Eigen::Vector2d p3 = U.row(F(k, 2));

		Eigen::Vector3d color1 = VertexColors.row(F(k, 0));
		Eigen::Vector3d color2 = VertexColors.row(F(k, 1));
		Eigen::Vector3d color3 = VertexColors.row(F(k, 2));


		Eigen::Vector2d transformedP1 = Eigen::Vector2d(p1(0) * 100, p1(1) * 100);
		Eigen::Vector2d transformedP2 = Eigen::Vector2d(p2(0) * 100, p2(1) * 100);
		Eigen::Vector2d transformedP3 = Eigen::Vector2d(p3(0) * 100, p3(1) * 100);


		myfile << "<polygon points = \"" << transformedP1(0) << "," << transformedP1(1) << " " << transformedP2(0) << "," << transformedP2(1) << " " << transformedP3(0) << "," << transformedP3(1) << "\" style = \"fill:rgb(" << color1(0) << "," << color1(1) << "," << color1(2) << ");stroke:rgb(" << color1(0) << "," << color1(1) << "," << color1(2) << ")\" />" << std::endl;
	}

	myfile << "</svg>" << std::endl;

	myfile.close();

}

void writeTextureFile(std::string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& VertexColors) {


	std::cout << "Writing texture file" << std::endl;

	int width = 50;
	int height = 50;

	Eigen::MatrixXd Colors(width * 3, height);

	std::cout << width * height * F.rows() << " iterations needed" << std::endl;

	for (unsigned int x = 0; x < width; x + 3) {
		std::cout << "Row " << x << " started" << std::endl;
		for (unsigned int y = 0; y < height; y++) {
			//get barycentric coordinates for every triangle
			for (unsigned int k = 0; k < F.rows(); k++) {

				Eigen::Vector3d p1 = V.row(F(k, 0));
				Eigen::Vector3d p2 = V.row(F(k, 1));
				Eigen::Vector3d p3 = V.row(F(k, 2));

				Eigen::Vector3d color1 = VertexColors.row(F(k, 0));
				Eigen::Vector3d color2 = VertexColors.row(F(k, 1));
				Eigen::Vector3d color3 = VertexColors.row(F(k, 2));

				float x_u = ((float)x / 3) / (float)width;
				float y_v = y / (float)height;

				float u, v, w;

				Barycentric(Eigen::Vector3d(x_u, y_v, 0), p1, p2, p3, u, v, w);

				if (u < 0 || v < 0 || w < 0) {
					continue;
				}

				Colors(x, y) = u * color1(0) + v * color2(0) + w * color3(0);
				Colors(x + 1, y) = u * color1(1) + v * color2(1) + w * color3(1);
				Colors(x + 2, y) = u * color1(2) + v * color2(2) + w * color3(2);
			}
		}
	}

	std::string texFileName = filename.substr(0, filename.find_last_of(".")) + ".png";

	stbi_write_png(texFileName.c_str(), width, height, 3, Colors.data(), width * 3);
}

void writeObj(std::string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N, Eigen::MatrixXd& Col)
{
	std::ofstream myfile;
	myfile.open(filename);
	myfile << "o " << filename << std::endl;
	for (int i = 0; i < V.rows(); i++)
	{
		//myfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
		myfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " " << Col(i, 0) << " " << Col(i, 1) << " " << Col(i, 2) << std::endl;
	}
	for (int i = 0; i < U.rows(); i++)
	{
		myfile << "vt " << U(i, 0) << " " << U(i, 1) << std::endl;
	}

	for (int i = 0; i < N.rows(); i++)
	{
		myfile << "vn " << N(i, 0) << " " << N(i, 1) << " " << N(i, 2) << std::endl;
	}

	for (int i = 0; i < F.rows(); i++)
	{
		myfile << "f " << F(i, 0) + 1 << "/" << F(i, 0) + 1 << "/" << F(i, 0) + 1 << " " << F(i, 1) + 1 << "/" << F(i, 1) + 1 << "/" << F(i, 1) + 1 << " " << F(i, 2) + 1 << "/" << F(i, 2) + 1 << "/" << F(i, 2) + 1 << std::endl;
	}


	//Create .mtl file reference
	std::string mtlFileName = filename.substr(0, filename.find_last_of(".")) + ".mtl";
	//myfile << "mtllib " << mtlFileName << std::endl;

	myfile.close();

	return;

	//Create .mtl file
	myfile.open(mtlFileName);
	myfile << "newmtl material_0" << std::endl;
	myfile << "Ka 1.000000 1.000000 1.000000" << std::endl;
	myfile << "Kd 1.000000 1.000000 1.000000" << std::endl;
	myfile << "Ks 0.000000 0.000000 0.000000" << std::endl;
	myfile << "d 1.0" << std::endl;
	myfile << "illum 2" << std::endl;
	myfile << "Ns 0.000000" << std::endl;
	myfile << "map_Kd " << filename.substr(0, filename.find_last_of(".")) << ".png" << std::endl;
	myfile.close();

}
/*
int main() {
	cv::Mat cameraMatrix, distanceCoefficients;
	cv::VideoCapture calibrationVideo(CALIBRATION_VIDEO_NAME), reconstructionVideo(RECONSTRUCTION_VIDEO_NAME);
	cv::aruco::ArucoDetector detector = createDetector();
	cv::aruco::Board* board = createBoard();

	if (RUN_CAMERA_CALIBRATION)
	{
		if (!videoExists(calibrationVideo))
		{
			return 0;
		}

		calibrateCamera(calibrationVideo, &detector, board, &cameraMatrix, &distanceCoefficients);
	}

	if (RUN_POSE_ESTIMATION_TEST)
	{
		if (!videoExists(reconstructionVideo))
		{
			return 0;
		}

		while (reconstructionVideo.grab())
		{
			cv::Mat image;
			reconstructionVideo.retrieve(image);

			Camera frame = Camera(image, cameraMatrix);
			Pose currentPose = frame.estimateCameraPose(&detector, board, cameraMatrix, distanceCoefficients);

			std::cout << "Camera translation vector: " << currentPose.position.x() << ", "
				<< currentPose.position.y() << ", " << currentPose.position.z() << std::endl;
			std::cout << "Camera rotation vector: " << currentPose.rotation.x() << ", "
				<< currentPose.rotation.y() << ", " << currentPose.rotation.z() << std::endl;
		}
	}

	if (RUN_VOXEL_GRID_TEST) {
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

	if (RUN_VOXEL_CARVING)
	{
		if (!videoExists(reconstructionVideo))
		{
			return 0;
		}

		std::vector<Camera> cameraFrames;
		int numFrames = reconstructionVideo.get(cv::CAP_PROP_FRAME_COUNT);
		auto grid = VoxelGrid::CreateFilledVoxelGrid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3i(50, 50, 50), 1);

		for (int i = 0; i < NUM_PROCESSED_FRAMES; i++)
		{
			cv::Mat image;
			reconstructionVideo.set(1, i * (numFrames / NUM_PROCESSED_FRAMES));
			reconstructionVideo.retrieve(image);

			cameraFrames.push_back(Camera(image, cameraMatrix));
		}

		SpaceCarver::MultiSweep(grid, cameraFrames);
		VoxelGridExporter::ExportToOFF(voxeTestFilenameTarget, grid);
	}

	if (RUN_UV_TEST) {
		Eigen::MatrixXd V, U_tutte, U, N;
		Eigen::MatrixXi F;

		igl::read_triangle_mesh(uvTestingInput, V, F);

		TutteEmbedder::GenerateUvMapping(V, F, U, N);

		igl::opengl::glfw::Viewer viewer;

		viewer.data().set_mesh(V, F);
		viewer.data().set_colors(N.array() * 0.5 + 0.5);
		viewer.data().show_texture = false;
		viewer.data().show_lines = false;

		Eigen::MatrixXd colors = Eigen::MatrixXd::Random(V.rows(), 3);
		colors = (colors + Eigen::MatrixXd::Constant(V.rows(), 3, 1.)) * 255 / 2.;

		writeObj("../beetleOut.obj", V, F, U_tutte, N);
		//writeTextureFile("../beetleOut.png", V, F, colors);
		writeSvgFile("../beetleOut.svg", V, F, colors);


		viewer.launch();
	}

	return 0;
}
 */

void tutte(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& U)
{
	Eigen::VectorXi bL;
	igl::boundary_loop(F, bL);

	Eigen::MatrixXd UV;
	igl::map_vertices_to_circle(V, bL, UV);

	Eigen::SparseMatrix<double> L(V.rows(), V.rows());
	igl::cotmatrix(V, F, L);

	igl::min_quad_with_fixed_data<double> data;
	igl::min_quad_with_fixed_precompute(L, bL, Eigen::SparseMatrix<double>(), false, data);

	Eigen::VectorXd B = Eigen::VectorXd::Zero(data.n, 1);
	igl::min_quad_with_fixed_solve(data, B, UV, Eigen::MatrixXd(), U);
	U.col(0) = -U.col(0);
}


int main(int argc, char* argv[])
{
	// Load input meshes
	Eigen::MatrixXd V, U_tutte, U;
	Eigen::MatrixXi F;

	igl::read_triangle_mesh(
		"../bunny.off", V, F);


	tutte(V, F, U_tutte);

	// Fit parameterization in unit sphere
	const auto normalize = [](Eigen::MatrixXd& U)
	{
		U.rowwise() -= U.colwise().mean().eval();
		U.array() /=
			(U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff() / 2.0;
	};


	const auto normalizeZeroToOne = [](Eigen::MatrixXd& U)
	{
		U.rowwise() -= U.colwise().minCoeff().eval();
		U.array() /=
			(U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff();
	};

	const auto normalizeMinusOneToOne = [](Eigen::MatrixXd& U)
	{
		U.rowwise() -= U.colwise().minCoeff().eval();
		U.array() /=
			(U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff() / 2.0;
		U.array() -= 1.0;
	};

	//normalizeZeroToOne(V);
	//normalizeZeroToOne(U_tutte);

	normalizeMinusOneToOne(V);
	normalizeMinusOneToOne(U_tutte);

	printf("Min and max: \n");


	U = U_tutte;
	Eigen::MatrixXd N;
	igl::per_vertex_normals(V, F, N);


	Eigen::MatrixXd colors = Eigen::MatrixXd::Random(V.rows(), 3);
	colors = (colors + Eigen::MatrixXd::Constant(V.rows(), 3, 1.)) / 2.;

	writeObj("../beetleOut.obj", V, F, U_tutte, N, colors);

	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1024, 768, "Texture Render", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
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

	/*
	static const GLfloat g_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 0.0f,  1.0f, 0.0f,
	};

	static const GLfloat g_color_buffer_data[] = {
		1.0f, 0.0f, 0.0f, // red
		1.0f, 1.0f, 0.0f, // green
		0.0f, 0.0f, 1.0f, // blue
	};
	*/

	std::vector<GLfloat> g_vertex_buffer_data;
	std::vector<GLfloat> g_color_buffer_data;

	for (int i = 0; i < F.rows(); i++) {
		//Get the three vertices of the face
		for (int j = 0; j < 3; j++)
		{
			int vertexIndex = F(i, j);
			float u = U(vertexIndex, 0);
			float v = U(vertexIndex, 1);
			g_vertex_buffer_data.push_back(u);
			g_vertex_buffer_data.push_back(v);
			g_vertex_buffer_data.push_back(0.0f);

			g_color_buffer_data.push_back(colors(vertexIndex, 0));
			g_color_buffer_data.push_back(colors(vertexIndex, 1));
			g_color_buffer_data.push_back(colors(vertexIndex, 2));
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

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight); /* Get size, store into specified variables  */

	/* Prepare the targa header */
	memcpy(tga.head, tgahead, 12);
	tga.dx = screenWidth;
	tga.dy = screenHeight;
	tga.head2 = 0x2018;

	std::cout << "creating texture file" << std::endl;

	/* Store pixels into tga.pic */
	glReadPixels(0, 0, screenWidth, screenHeight, GL_RGB, GL_UNSIGNED_BYTE, tga.pic[0]);

	std::string filename = "screenshot.png";

	stbi_write_png(filename.c_str(), screenWidth, screenHeight, 3, tga.pic[0], screenWidth * 3);

	std::cout << "texture file written" << std::endl;

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return EXIT_SUCCESS;
}

