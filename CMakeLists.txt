cmake_minimum_required(VERSION 3.13)

project(voxel-carving)

# Options
set(CMAKE_CXX_STANDARD 14)
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv-build/cmake CACHE PATH "Path to installed OpenCV")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(UTILS
    utils/utils.h
)

set(HEADER_FILES 
    Camera.h
    voxel/SpaceCarver.h
    voxel/VoxelGrid.h
    voxel/VoxelGridExporter.h
	voxel/SimpleMesh.h
	voxel/MarchingCubes.h
)

add_executable(voxel-carving ${UTILS} ${HEADER_FILES} main.cpp)

target_include_directories(voxel-carving PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(voxel-carving PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(voxel-carving Eigen3::Eigen)
target_link_libraries(voxel-carving ${OpenCV_LIBS})

if(WIN32)
    target_compile_definitions(voxel-carving PUBLIC NOMINMAX _USE_MATH_DEFINES)
    set_property(TARGET voxel-carving PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
endif(WIN32)
