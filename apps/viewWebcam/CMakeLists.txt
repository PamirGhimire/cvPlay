file(GLOB SOURCE_FILES_VIEW_WEBCAM 
	viewWebcam.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(viewWebcam 
	${SOURCE_FILES_VIEW_WEBCAM} 
)

target_link_libraries (viewWebcam 
	${OpenCV_LIBS}
)