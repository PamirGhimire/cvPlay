file(GLOB SOURCE_FILES_MONO_STREAM_RIGID_TRANSFORMATION_ESTIMATION 
	*.cpp
	${PROJECT_SOURCE_DIR}/vision/*.cpp
)

add_executable(monoStreamRigidTransformationEstimation 
	${SOURCE_FILES_MONO_STREAM_RIGID_TRANSFORMATION_ESTIMATION} 
)

target_link_libraries (monoStreamRigidTransformationEstimation 
	${OpenCV_LIBS}
)