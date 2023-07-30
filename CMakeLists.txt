cmake_minimum_required(VERSION 3.13)

project(voxel-carving)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Compile external dependencies 
add_subdirectory (external)

# Options
set(CMAKE_CXX_STANDARD 14)
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv-build/cmake CACHE PATH "Path to installed OpenCV")

find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

# Libigl
include(libigl)

include_directories(
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	.
)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(UTILS
    utils/utils.h
	volume.cpp
)


set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)

set(HEADER_FILES 
    Camera.h
    voxel/SpaceCarver.h
    voxel/VoxelGrid.h
    voxel/VoxelGridExporter.h
	voxel/SimpleMesh.h
	voxel/MarchingCubes.h
	stb_image_write.h
	TutteEmbedding.h
	RBF.h
	MarchingCubes2.h
	Volume.h
	export.h
)

set(EXPORT_FILES
	export.cpp
)

add_executable(voxel-carving ${UTILS} ${HEADER_FILES} ${EXPORT_FILES} main.cpp)

target_include_directories(voxel-carving PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(voxel-carving PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(voxel-carving Eigen3::Eigen)
target_link_libraries(voxel-carving igl::core)
target_link_libraries(voxel-carving ${OpenCV_LIBS})
target_link_libraries(voxel-carving ${ALL_LIBS})

if(WIN32)
    target_compile_definitions(voxel-carving PUBLIC NOMINMAX _USE_MATH_DEFINES)
    set_property(TARGET voxel-carving PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
endif(WIN32)

SOURCE_GROUP(shaders REGULAR_EXPRESSION ".*/.*shader$" )

file(COPY TransformVertexShader.vertexshader DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ColorFragmentShader.fragmentshader DESTINATION ${CMAKE_BINARY_DIR})
