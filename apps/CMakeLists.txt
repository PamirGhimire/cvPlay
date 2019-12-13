set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin/apps/)

include_directories(
	${PROJECT_SOURCE_DIR}/vision/
)

add_subdirectory(viewWebcam)
add_subdirectory(viewORBInWebcam)
add_subdirectory(monocularStreamFeatureMatching)
add_subdirectory(monocularStreamFeatureMatching_multiThreaded)
add_subdirectory(monoStreamFundamentalMatEstimation)
#add_subdirectory(monoStreamFundamentalMatEstimation_multiThreaded) #needs fixing