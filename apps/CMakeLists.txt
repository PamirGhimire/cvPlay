set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/apps/*.cpp)

file(GLOB HEADER_FILES ${PROJECT_SOURCE_DIR}/apps/*.h)

include_directories(
	${PROJECT_SOURCE_DIR}/apps/
)

add_executable(testapp 
	${SOURCE_FILES} 
	${HEADER_FILES}
)

target_link_libraries (testapp 
	Eigen3::Eigen 
	${OpenCV_LIBS}
)

