set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin/apps/)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
	${PROJECT_SOURCE_DIR}/apps/
	${PROJECT_SOURCE_DIR}/common/
)

#-----------viewWebcam--app-------------------
file(GLOB SOURCE_FILES_VIEW_WEBCAM 
	${PROJECT_SOURCE_DIR}/apps/viewWebcam.cpp
	${PROJECT_SOURCE_DIR}/common/*.cpp
)

add_executable(viewWebcam 
	${SOURCE_FILES_VIEW_WEBCAM} 
)

target_link_libraries (viewWebcam 
	${OpenCV_LIBS}
)
#-----------viewWebcam--app-------------------


#-----------viewORBInWebcam--app-------------------
file(GLOB SOURCE_FILES_VIEW_ORB_IN_WEBCAM 
	${PROJECT_SOURCE_DIR}/apps/viewORBInWebcam.cpp
	${PROJECT_SOURCE_DIR}/common/*.cpp
)

add_executable(viewORBInWebcam 
	${SOURCE_FILES_VIEW_ORB_IN_WEBCAM} 
)
target_link_libraries (viewORBInWebcam 
	${OpenCV_LIBS}
)
#-----------viewORBInWebcam--app-------------------

#-----------monocularStreamFeatureMatching--app-------------------
file(GLOB SOURCE_FILES_MONOCULAR_FEATURE_MATCHING 
	${PROJECT_SOURCE_DIR}/apps/monocularStreamFeatureMatching.cpp
	${PROJECT_SOURCE_DIR}/common/*.cpp
)

add_executable(monocularStreamFeatureMatching 
	${SOURCE_FILES_MONOCULAR_FEATURE_MATCHING} 
)
target_link_libraries (monocularStreamFeatureMatching 
	${OpenCV_LIBS}
)
#-----------viewORBInWebcam--app-------------------