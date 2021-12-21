cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME "MLSDemo")

project(${PROJECT_NAME})




if (WIN32)
    if (DEFINED ENV{DEV_LIB_DIR})
        message("Found DEV_LIB_DIR=$ENV{DEV_LIB_DIR}")
    else()
        message(FATAL_ERROR "No Found DEV Library, Please specify DEV Library Path by environment variable DEV_LIB_DIR")
    endif()
    message("WINDOWS")
    include_directories(
        $ENV{DEV_LIB_DIR}/opencv/include
		$ENV{DEV_LIB_DIR}/spdlog/include
		$ENV{DEV_LIB_DIR}/jsoncpp/include
        $ENV{DEV_LIB_DIR}/libssh2/include
		
    )
    
    link_directories(
        $ENV{DEV_LIB_DIR}/opencv/lib
		$ENV{DEV_LIB_DIR}/spdlog/lib
		$ENV{DEV_LIB_DIR}/jsoncpp/lib
        $ENV{DEV_LIB_DIR}/libssh2/lib
    )
    
    link_libraries(
        opencv_world347
		spdlog
		jsoncpp
		utils
		
    )
else()
    message("UNIX")
endif()

aux_source_directory("src" SRC)

add_executable(${PROJECT_NAME} ${SRC})

