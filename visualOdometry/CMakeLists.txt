set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/visualSlam/)
set(CMAKE_CXX_STANDARD 11)

file(GLOB HEADER_FILES 
	${PROJECT_SOURCE_DIR}/visualSlam/include/*.h
)

file(GLOB SOURCE_FILES 
	${PROJECT_SOURCE_DIR}/visualSlam/src/*.cpp
)

include_directories(
	${PROJECT_SOURCE_DIR}/visualSlam/include/
)

add_executable(toyproject 
	${SOURCE_FILES} 
	${HEADER_FILES}
)

target_link_libraries (toyproject 
	Eigen3::Eigen 
	${OpenCV_LIBS}
)

