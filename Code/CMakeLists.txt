cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME AlgorithmDemo)

project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory("mls")