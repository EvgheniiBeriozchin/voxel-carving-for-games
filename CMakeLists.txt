cmake_minimum_required(VERSION 3.13)

project(voxel-carving)

# Options
set(CMAKE_CXX_STANDARD 14)
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")

find_package(Eigen3 REQUIRED)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(UTILS
    utils/utils.cpp
)

add_executable(voxel-carving ${UTILS} main.cpp)
target_link_libraries(voxel-carving Eigen3::Eigen)
target_include_directories(voxel-carving PUBLIC ${EIGEN3_INCLUDE_DIR})

if(WIN32)
    target_compile_definitions(voxel-carving PUBLIC NOMINMAX _USE_MATH_DEFINES)
    set_property(TARGET voxel-carving PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
endif(WIN32)
