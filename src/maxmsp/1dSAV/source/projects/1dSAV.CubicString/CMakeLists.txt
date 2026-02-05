# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

set(AUDIO_PROCESSORS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../processor)

#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}"
	${CMAKE_CURRENT_SOURCE_DIR}
	${AUDIO_PROCESSORS_DIR}
)


set( SOURCE_FILES
	${PROJECT_NAME}.cpp
	${AUDIO_PROCESSORS_DIR}/StringProcessor.cpp
)


find_package (Eigen3 REQUIRED NO_MODULE)

add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

if(APPLE)
	target_link_libraries(
		${PROJECT_NAME} PRIVATE
		"-framework Accelerate"
	)
endif()

target_link_libraries(
	${PROJECT_NAME} PUBLIC
	Eigen3::Eigen
)


include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)


#############################################################
# UNIT TEST
#############################################################

include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)
