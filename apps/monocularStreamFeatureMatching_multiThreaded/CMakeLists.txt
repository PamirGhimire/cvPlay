file(GLOB SOURCE_FILES_MONOCULAR_FEATURE_MATCHING_MULTITHREADED 
	monocularStreamFeatureMatching_multiThreaded.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(monocularStreamFeatureMatching_multiThreaded 
	${SOURCE_FILES_MONOCULAR_FEATURE_MATCHING_MULTITHREADED} 
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries (monocularStreamFeatureMatching_multiThreaded 
	${OpenCV_LIBS}
	Threads::Threads
)