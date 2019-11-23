project(visualSmal)

cmake_minimum_required(VERSION 3.1)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

#------EIGEN------------
set(Eigen3_DIR "/home/pamir/Desktop/eigen_install_dir/share/eigen3/cmake/")
find_package(Eigen3 REQUIRED CONFIG PATHS Eigen3_DIR)
message(STATUS "Eigen3 found, version ${Eigen3_VERSION}. ${EIGEN3_INCLUDE_DIRS}")
#------EIGEN------------

#------opencv------------
set(OpenCV_DIR "/home/pamir/Desktop/opencv-3.4.0/build/install_dir/share/OpenCV")
find_package(OpenCV REQUIRED CONFIG PATHS OpenCV_DIR)
message(STATUS "OpenCV found, version ${OpenCV_VERSION}")
#------opencv------------


include_directories(
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

add_subdirectory(apps)
add_subdirectory(common)
