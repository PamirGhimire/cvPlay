set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin/common)
set(CMAKE_CXX_STANDARD 11)

file(GLOB HEADER_FILES 
	${PROJECT_SOURCE_DIR}/common/*.h
)

file(GLOB SOURCE_FILES 
	${PROJECT_SOURCE_DIR}/common/*.cpp
	${PROJECT_SOURCE_DIR}/common/tests/*.cpp
)

include_directories(
	${PROJECT_SOURCE_DIR}/common/
)

add_executable(CameraSpec 
	${SOURCE_FILES} 
	${HEADER_FILES}
)

target_link_libraries (CameraSpec 
	${OpenCV_LIBS}
)

