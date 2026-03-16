cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME timeCubicString)
project(${PROJECT_NAME})

message("${CMAKE_SYSTEM_FRAMEWORK_PATH}")

set(AUDIO_PROCESSORS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../processor)

include_directories(
	${AUDIO_PROCESSORS_DIR}
)

set( SOURCE_FILES
    ${PROJECT_NAME}.cpp
	${AUDIO_PROCESSORS_DIR}/StringProcessor.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

if(APPLE)
	target_link_libraries(
		${PROJECT_NAME} PRIVATE
		"-framework Accelerate"
	)
endif()

# Use Eigen for the processor (or comment if not wanted)
add_compile_definitions(EIGENPROC)

find_package (Eigen3 REQUIRED NO_MODULE)
target_link_libraries(
	${PROJECT_NAME} PUBLIC
	Eigen3::Eigen
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Optimisation flags
if(NOT MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
			-O3
			-march=native
			-ffast-math
			-funroll-loops
			-fno-omit-frame-pointer
	)
	target_compile_definitions(${PROJECT_NAME} PRIVATE
			EIGEN_NO_DEBUG
			EIGEN_DONT_PARALLELIZE
	)
endif()
