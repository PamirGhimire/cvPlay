file(GLOB SOURCE_FILES_MONO_STREAM_FUND_MAT_EST_MULTITHREADED 
	monoStreamFundamentalMatEstimation_multiThreaded.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(monoStreamFundamentalMatEstimation_multiThreaded 
	${SOURCE_FILES_MONO_STREAM_FUND_MAT_EST_MULTITHREADED} 
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries (monoStreamFundamentalMatEstimation_multiThreaded 
	${OpenCV_LIBS}
	Threads::Threads
)