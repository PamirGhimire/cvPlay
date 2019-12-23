file(GLOB SOURCE_FILES_MONOCULAR_FEATURE_MATCHING 
	monocularStreamFeatureMatching.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(monocularStreamFeatureMatching 
	${SOURCE_FILES_MONOCULAR_FEATURE_MATCHING} 
)
target_link_libraries (monocularStreamFeatureMatching 
	${OpenCV_LIBS}
)