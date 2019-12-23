file(GLOB SOURCE_FILES_MONO_STREAM_FUNDAMENTAL_MAT_ESTIMATION 
	monoStreamFundamentalMatEstimation.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(monoStreamFundamentalMatEstimation 
	${SOURCE_FILES_MONO_STREAM_FUNDAMENTAL_MAT_ESTIMATION} 
)

target_link_libraries (monoStreamFundamentalMatEstimation 
	${OpenCV_LIBS}
)