file(GLOB SOURCE_FILES_VIEW_ORB_IN_WEBCAM 
	viewORBInWebcam.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(viewORBInWebcam 
	${SOURCE_FILES_VIEW_ORB_IN_WEBCAM} 
)
target_link_libraries (viewORBInWebcam 
	${OpenCV_LIBS}
)