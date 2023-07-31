#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
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


#include <opencv2/opencv.hpp>

#include "voxel/VoxelGrid.h"
#include "voxel/VoxelGridExporter.h"
#include "voxel/SpaceCarver.h"
#include "Camera.h"
#include "utils/utils.h"
#include "TutteEmbedding.h"

#include <CGAL/boost/graph/iterator.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/surface_mesh_parameterization.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <voxel/SimpleMesh.h>
#include <voxel/MarchingCubes.h>

#include <boost/function_output_iterator.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/foreach.hpp>


typedef CGAL::Simple_cartesian<double>           Kernel;
typedef Kernel::Point_2                         Point_2;
typedef Kernel::Point_3                         Point_3;
typedef CGAL::Surface_mesh<Kernel::Point_3>     SurfaceMesh;

typedef boost::graph_traits<SurfaceMesh>::vertex_descriptor     vertex_descriptor;
typedef boost::graph_traits<SurfaceMesh>::halfedge_descriptor   halfedge_descriptor;
typedef boost::graph_traits<SurfaceMesh>::face_descriptor       face_descriptor;

typedef SurfaceMesh::Property_map<vertex_descriptor, Point_2>  UV_pmap;

typedef Kernel::Vector_3                                               Vector;
typedef Kernel::Compare_dihedral_angle_3                    Compare_dihedral_angle_3;


#define RUN_CAMERA_CALIBRATION 1
#define RUN_POSE_ESTIMATION_TEST 0
#define RUN_VOXEL_GRID_TEST 0
#define RUN_VOXEL_CARVING 1
#define RUN_CAMERA_ESTIMATION_EXPORT 0
#define RUN_TUTTE_EMBEDDING_TEST 0
#define EXPORT_TEXTURED_MESH 1


const int NUM_PROCESSED_FRAMES = 25;
const std::string CALIBRATION_VIDEO_NAME = "../PepperMill_NaturalLight.mp4";
const std::string RECONSTRUCTION_VIDEO_NAME = "../PepperMill_CameraLight.mp4";
//const std::string RECONSTRUCTION_VIDEO_NAME = "../Box_NaturalLight.mp4";
const std::string voxeTestFilenameTarget = std::string("voxelGrid.off");
const std::string uvTestingInput = "../bunny.off";

typedef struct
{
	unsigned char head[12];
	unsigned short dx /* Width */, dy /* Height */, head2;
	unsigned char pic[768 * 1024 * 10][3];
} typetga;

static typetga tga;

unsigned char const tgahead[12] = { 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

template <typename G>
struct Constraint : public boost::put_get_helper<bool, Constraint<G> >
{
	typedef typename boost::graph_traits<G>::edge_descriptor edge_descriptor;
	typedef boost::readable_property_map_tag      category;
	typedef bool                                  value_type;
	typedef bool                                  reference;
	typedef edge_descriptor                       key_type;
	Constraint()
		:g_(NULL)
	{}
	Constraint(G& g, double bound)
		: g_(&g), bound_(bound)
	{}
	bool operator[](edge_descriptor e) const
	{
		const G& g = *g_;
		return compare_(g.point(source(e, g)),
			g.point(target(e, g)),
			g.point(target(next(halfedge(e, g), g), g)),
			g.point(target(next(opposite(halfedge(e, g), g), g), g)),
			bound_) == CGAL::SMALLER;
	}
	const G* g_;
	Compare_dihedral_angle_3 compare_;
	double bound_;
};

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

void RenderTexture(std::string textureName, Eigen::MatrixXd& U, Eigen::MatrixXi& F, Eigen::MatrixXd& Colors)
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

	std::vector<GLfloat> g_vertex_buffer_data;
	std::vector<GLfloat> g_color_buffer_data;

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

void WriteObj(std::string objName, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& U, Eigen::MatrixXd& N, Eigen::MatrixXd& Col)
{
	std::cout << "Writing OBJ file " << objName << ".obj" << std::endl;
	std::ofstream myfile;
	std::string filename = objName + ".obj";


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
		myfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
	}

	//Write texture coordinates
	for (int i = 0; i < U.rows(); i++)
	{
		myfile << "vt " << U(i, 0) << " " << U(i, 1) << std::endl;
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

		int numFrames = reconstructionVideo.get(cv::CAP_PROP_FRAME_COUNT);
		for (int i = 0; i < NUM_PROCESSED_FRAMES; i++)
		{
			reconstructionVideo.set(1, i * (numFrames / NUM_PROCESSED_FRAMES));
			cv::Mat image;
			reconstructionVideo.retrieve(image);

			Camera frame = Camera(image, cameraMatrix);
			Eigen::Matrix4d currentPose = frame.estimateCameraPose(&detector, board, cameraMatrix, distanceCoefficients);

			std::cout << "Camera pose: " << currentPose << std::endl;
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


		Eigen::Vector3d boardCenter = 0.01 * Eigen::Vector3d(10.5, 14.25, 0.0);
		Eigen::Vector3d gridOrigin = 0.01 * Eigen::Vector3d(3.5, 3.5, 0);
		// Real dimension cm
		double xSizeCM = 14;
		double ySizeCM = 21.5;
		double zSizeCM = 10;
		// VoxelDimension

		double voxelPerCM = 2;
		double xSizeVX = xSizeCM * voxelPerCM;
		double ySizeVX = ySizeCM * voxelPerCM;
		double zSizeVX = zSizeCM * voxelPerCM;
		double voxelSize = 0.01 / voxelPerCM;
		auto grid = VoxelGrid::CreateFilledVoxelGrid(gridOrigin, Eigen::Vector3i(xSizeVX, ySizeVX, zSizeVX), voxelSize);
		//auto grid = VoxelGrid::CreateFilledVoxelGrid(gridOrigin, Eigen::Vector3i(3, 3, 5), 0.05);

		std::cout << "Preparing frames for voxel carving" << std::endl;
		for (int i = 0; i < NUM_PROCESSED_FRAMES; i++)
		{
			if ((i + 1) % 10 == 0)
			{
				std::cout << "Processed " << (i + 1) << " frames" << std::endl;
			}
			cv::Mat image;
			reconstructionVideo.set(1, i * (numFrames / NUM_PROCESSED_FRAMES));
			reconstructionVideo.retrieve(image);

			Camera cam = Camera(image, cameraMatrix);
			cam.pose = cam.estimateCameraPose(&detector, board, cameraMatrix, distanceCoefficients);
			cameraFrames.push_back(cam);
		}



		// Camera To markers output
		if (RUN_CAMERA_ESTIMATION_EXPORT) {
			int ci = 0;
			std::vector<std::vector<Eigen::Vector3d>> points;
			points.push_back(std::vector<Eigen::Vector3d>());
			std::vector<Eigen::Vector3d> colors;
			colors.push_back(Eigen::Vector3d(0, 255, 255));
			for (int i = 0; i < cameraFrames.size(); i++)
			{
				auto pos = cameraFrames[i].pose.block<3, 1>(0, 3);
				points[ci].push_back(pos);
			}
			// x axis
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(255, 0, 0));
			for (int i = -100; i < 100; i += 2) {
				points[ci].push_back(0.01 * Eigen::Vector3d(i, 0, 0));
			}
			// y axis
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(0, 255, 0));
			for (int i = -100; i < 100; i += 2) {
				points[ci].push_back(0.01 * Eigen::Vector3d(0, i, 0));
			}
			// z axis
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(0, 0, 255));
			for (int i = 0; i < 100; i += 2) {
				points[ci].push_back(0.01 * Eigen::Vector3d(0, 0, i));
			}
			// marker setup
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(0, 0, 0));
			points[ci].push_back(boardCenter);
			for (int i = 0; i < boardCenter.x() * 2; i++) {
				points[ci].push_back(0.01 * Eigen::Vector3d(i, 0, 0));
				points[ci].push_back(Eigen::Vector3d(0.01 * i, boardCenter.y() * 2, 0));
			}
			for (int i = 0; i < boardCenter.y() * 2; i++) {
				points[ci].push_back(0.01 * Eigen::Vector3d(0, i, 0));
				points[ci].push_back(Eigen::Vector3d(boardCenter.x() * 2, 0.01 * i, 0));
			}
			ci++;
			points.push_back(std::vector<Eigen::Vector3d>());
			colors.push_back(Eigen::Vector3d(255, 255, 255));
			for (int i = 0; i < cameraFrames.size(); i++)
			{
				for (int j = 0; j < cameraFrames[i].objectPoints.size(); j++) {
					points[ci].push_back(Eigen::Vector3d(
						cameraFrames[i].objectPoints[j].x,
						cameraFrames[i].objectPoints[j].y,
						cameraFrames[i].objectPoints[j].z
					));
				}
			}

			VoxelGridExporter::ExportToPLY("cameraPoses.ply", points, colors);
			VoxelGridExporter::ExportToOFF("voxelGrid_cameraPoses.off", grid);
		}
		int count = 0;
		for (int i = 0; i < cameraFrames.size(); i++)
		{
			Eigen::Vector2i projection = cameraFrames[i].ProjectIntoCameraSpace(Eigen::Vector3d(0, 0, 0));
			if (projection.x() >= 0 && projection.x() < cameraFrames[i].frame.size().width
				&& projection.y() >= 0 && projection.y() < cameraFrames[i].frame.size().height)
			{
				count++;
				std::cout << "World center location: " << projection << std::endl;
			}
		}
		std::cout << "Center points within frame: " << count << std::endl;

		// write image with projected grid positions
		cv::Mat tf;
		int index = 0;

		for (/*auto cameraFrame: cameraFrames*/int i = 0; i < 1; i++)
		{
			//if (index > 10)
			//	break;
			auto cameraFrame = cameraFrames[i];
			cameraFrame.frame.copyTo(tf);
			for each (auto v in grid.GetBoundaryVoxels())
			{
				Eigen::Vector3d v2 = grid.GetVoxelCenter(v);
				auto pixelPos = cameraFrame.ProjectIntoCameraSpace(v2);
				if (pixelPos.x() >= tf.cols || pixelPos.y() >= tf.rows || pixelPos.x() < 0 || pixelPos.y() < 0)
					continue;
				cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(0, 0, 255), 5);
			}
			auto pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0, 0, 0));
			//std::cout << "00 pixel: " << pixelPos << std::endl;
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(255, 255, 0), 5);
			//std::cout << "010 pixel: " << pixelPos << std::endl;
			pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0.1, 0, 0));
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(0, 0, 255), 5);
			//std::cout << "001 pixel: " << pixelPos << std::endl;
			pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0, 0.1, 0));
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(0, 255, 0), 5);
			//std::cout << "0101 pixel: " << pixelPos << std::endl;
			pixelPos = cameraFrame.ProjectIntoCameraSpace(Eigen::Vector3d(0, 0, 0.1));
			cv::circle(tf, cv::Point(pixelPos.x(), pixelPos.y()), 5, cv::Vec3b(255, 0, 0), 5);
			cv::imwrite("camera_" + std::to_string(index++) + "_gray.png", tf);
		}
		std::cout << "Running voxel carving" << std::endl;
		SpaceCarver::MultiSweep(grid, cameraFrames);
		VoxelGridExporter::ExportToOFF(voxeTestFilenameTarget, grid);


		std::cout << "Creating zero enclosed voxel grid" << std::endl;
		VoxelGrid enclosedGrid = VoxelGrid::GetZeroEnclosedVoxelGrid(grid);

		std::cout << "Creating Mesh" << std::endl;
		SimpleMesh SIMPLEmesh;
		CreateMesh(&enclosedGrid, &SIMPLEmesh);
		std::cout << "Mesh created" << std::endl;
		std::cout << "Vertices: " << SIMPLEmesh.GetVertices().size() << std::endl;
		std::cout << "Faces: " << SIMPLEmesh.GetTriangles().size() << std::endl;


		if (EXPORT_TEXTURED_MESH) {
			
			// Load input meshes
			Eigen::MatrixXd V, U, N;
			Eigen::MatrixXi F, C;

			SIMPLEmesh.WriteMesh("mesh.off");

			SIMPLEmesh.DeduplicateVertices();

			SIMPLEmesh.WriteMesh("meshSimple.off");

			std::vector<Vertex> m_vertices = SIMPLEmesh.GetVertices();
			std::vector<Triangle> m_triangles = SIMPLEmesh.GetTriangles();
			std::vector<cv::Vec3b> m_colors = SIMPLEmesh.GetColors();


			std::cout << "Creating CGAL mesh" << std::endl;
			SurfaceMesh sm;

			

			SurfaceMesh::Property_map<vertex_descriptor, unsigned int> indexMap = sm.add_property_map<vertex_descriptor, unsigned int>("v:index", -1).first;


			//Load vertices, faces and colors into CGAL mesh
			for (unsigned int i = 0; i < m_vertices.size(); i++)
			{
				Point_3 p(m_vertices[i].x(), m_vertices[i].y(), m_vertices[i].z());
				SurfaceMesh::vertex_index face_index = sm.add_vertex(p);
				indexMap[face_index] = i;
			}



			//face index map

			for (unsigned int i = 0; i < m_triangles.size(); i++)
			{
				SurfaceMesh::face_index face_index = sm.add_face(
					SurfaceMesh::Vertex_index(m_triangles[i].idx0),
					SurfaceMesh::Vertex_index(m_triangles[i].idx1),
					SurfaceMesh::Vertex_index(m_triangles[i].idx2));
			}
			

			//Add color property
			sm.add_property_map<vertex_descriptor, Color>("v:color", Color(0, 0, 0)).first;
			auto colorMap = sm.property_map<vertex_descriptor, Color>("v:color").first;

			

			for (unsigned int i = 0; i < m_colors.size(); i++)
			{
				colorMap[SurfaceMesh::Vertex_index(i)] = Color(m_colors[i][2], m_colors[i][1], m_colors[i][0]);
			}

			std::cout << "CGAL mesh created" << std::endl;

			std::cout << "Duplicating non-manifold vertices" << std::endl;
			
			std::cout << "Vertices before: " << sm.number_of_vertices() << std::endl;
			CGAL::Polygon_mesh_processing::duplicate_non_manifold_vertices(sm);

			std::cout << "Vertices after: " << sm.number_of_vertices() << std::endl;

			// Remove degenerate faces
;

			//int removed = CGAL::Polygon_mesh_processing::keep_largest_connected_components(sm, 1, fi);

			std::cout << "Vertices after removing components: " << sm.number_of_vertices() << std::endl;

			std::cout << "Removing degenerate faces" << std::endl;

			CGAL::Polygon_mesh_processing::remove_almost_degenerate_faces(sm);

			std::cout << "Vertices after removing degernate faces: " << sm.number_of_vertices() << std::endl;

			std::cout << "Removed degenerate faces" << std::endl;

			halfedge_descriptor bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;

			// The UV property map that holds the parameterized values
			typedef SurfaceMesh::Property_map<vertex_descriptor, Point_2>  UV_pmap;
			UV_pmap uv_map = sm.add_property_map<vertex_descriptor, Point_2>("h:uv").first;

			CGAL::Surface_mesh_parameterization::parameterize(sm, bhd, uv_map);

			std::cout << "Parameterized" << std::endl;

			// Compute normals

			std::cout << "Computing normals" << std::endl;

			SurfaceMesh::Property_map<vertex_descriptor, Vector> vnormals = sm.add_property_map<vertex_descriptor, Vector>("v:normals", CGAL::NULL_VECTOR).first;

			CGAL::Polygon_mesh_processing::compute_vertex_normals(sm, vnormals);

			std::cout << "Computed normals" << std::endl;

			std::cout << "Creating output format" << std::endl;

			//Copy V, F, and U from sm
			V.resize(sm.number_of_vertices(), 3);
			F.resize(sm.number_of_faces(), 3);

			int i = 0;
			for (SurfaceMesh::Vertex_index v : sm.vertices()) {
				Point_3 p = sm.point(v);
				V.row(i) = Eigen::Vector3d(p.x(), p.y(), p.z());
				i++;
			}

			F.resize(sm.number_of_faces(), 3);
			i = 0;
			
			for (SurfaceMesh::Face_index f : sm.faces()) {
				SurfaceMesh::Halfedge_index hf = sm.halfedge(f);

				std::vector<SurfaceMesh::Vertex_index> face_vertices;
				for (SurfaceMesh::Halfedge_index hi : CGAL::halfedges_around_face(hf, sm)) {
					SurfaceMesh::Vertex_index vi = sm.target(hi);
					face_vertices.push_back(vi);
				}

				bool addFace = true;
				//check if any vertex indices are larger than the number of vertices
				for (int i = 0; i < 3; i++) {
					if (face_vertices[i] >= sm.number_of_vertices()) {
						std::cout << "Vertex index " << face_vertices[i] << " is larger than the number of vertices " << sm.number_of_vertices() << std::endl;
						addFace = false;
					}
				}

				if (!addFace) {
					continue;
				}
				F.row(i) = Eigen::Vector3i(face_vertices[0], face_vertices[1], face_vertices[2]);
				i++;
			}
			
			
			/*
			std::vector<unsigned>* faces = new std::vector<unsigned>[sm.number_of_faces()];
			std::cout << "Iterate over faces\n";
			{
				unsigned i_face = 0;
				BOOST_FOREACH(face_descriptor fd, sm.faces()) {
					BOOST_FOREACH(vertex_descriptor vd, vertices_around_face(sm.halfedge(fd), sm)) {
						faces[i_face].push_back(vd);
					}
					F(i_face, 0) = faces[i_face][0];
					F(i_face, 1) = faces[i_face][1];
					F(i_face, 2) = faces[i_face][2];

					i_face++;

				}
			}
			*/
			

			U.resize(sm.number_of_vertices(), 2);
			auto vertices = sm.vertices();
			auto v = vertices.first;
			i = 0;
			for (int i = 0; i < sm.number_of_vertices(); i++) {
				U(i, 0) = uv_map[*v].x();
				U(i, 1) = uv_map[*v].y();

				v++;
			}

			N.resize(sm.number_of_vertices(), 3);
			v = vertices.first;
			i = 0;
			for (int i = 0; i < sm.number_of_vertices(); i++) {
				N(i, 0) = vnormals[*v].x();
				N(i, 1) = vnormals[*v].y();
				N(i, 2) = vnormals[*v].z();

				v++;
			}

			Eigen::MatrixXd colors = Eigen::MatrixXd::Random(V.rows(), 3);
			//colors = (colors + Eigen::MatrixXd::Constant(V.rows(), 3, 1.)) / 2.;

			//Remap Colors from [0,255] to [0,1]
			colors = colors / 255.;

			std::cout << "Created output format" << std::endl;

			WriteObj("mesh", V, F, U, N, colors);

			RenderTexture("mesh", U, F, colors);

			return 0;
			
		}

		SIMPLEmesh.WriteMesh("mesh.off");

		
	}

	if (RUN_TUTTE_EMBEDDING_TEST) {
		// Load input meshes
		Eigen::MatrixXd V, U, N;
		Eigen::MatrixXi F;

		//igl::read_triangle_mesh(uvTestingInput, V, F);

		SurfaceMesh sm;

		CGAL::IO::read_polygon_mesh(uvTestingInput, sm);

		// CGAL::Polygon_mesh_processing::remove_almost_degenerate_faces(sm);

		//TutteEmbedder::GenerateUvMapping(sm, V, F, U, N);

		halfedge_descriptor bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;

		// The UV property map that holds the parameterized values
		typedef SurfaceMesh::Property_map<vertex_descriptor, Point_2>  UV_pmap;
		UV_pmap uv_map = sm.add_property_map<vertex_descriptor, Point_2>("h:uv").first;

		CGAL::Surface_mesh_parameterization::parameterize(sm, bhd, uv_map);

		SurfaceMesh::Property_map<vertex_descriptor, Vector> vnormals = sm.add_property_map<vertex_descriptor, Vector>("v:normals", CGAL::NULL_VECTOR).first;

		CGAL::Polygon_mesh_processing::compute_vertex_normals(sm, vnormals);


		
		//Copy V, F, and U from sm
		V.resize(sm.number_of_vertices(), 3);
		F.resize(sm.number_of_faces(), 3);

		int i = 0;
		for (auto v : sm.vertices()) {
			auto p = sm.point(v);
			V.row(i) = Eigen::Vector3d(p.x(), p.y(), p.z());
			i++;
		}

		for (SurfaceMesh::Face_index f : sm.faces()) {
			CGAL::Vertex_around_face_circulator<SurfaceMesh> vcirc(sm.halfedge(f), sm), done(vcirc);
			//get the next three vertices and store them in the face matrix
			F(f.idx(), 0) = vcirc->idx(); ++vcirc;
			F(f.idx(), 1) = vcirc->idx(); ++vcirc;
			F(f.idx(), 2) = vcirc->idx();
		}

		U.resize(sm.number_of_vertices(), 2);
		auto vertices = sm.vertices();
		auto v = vertices.first;
		i = 0;
		for (int i = 0; i < sm.number_of_vertices(); i++) {
			U(i, 0) = uv_map[*v].x();
			U(i, 1) = uv_map[*v].y();

			v++;
		}

		N.resize(sm.number_of_vertices(), 3);
		v = vertices.first;
		i = 0;
		for (int i = 0; i < sm.number_of_vertices(); i++) {
			N(i, 0) = vnormals[*v].x();
			N(i, 1) = vnormals[*v].y();
			N(i, 2) = vnormals[*v].z();

			v++;
		}

		Eigen::MatrixXd colors = Eigen::MatrixXd::Random(V.rows(), 3);
		colors = (colors + Eigen::MatrixXd::Constant(V.rows(), 3, 1.)) / 2.;

		WriteObj("mesh", V, F, U, N, colors);

		RenderTexture("mesh", U, F, colors);
	}

	return 0;
}