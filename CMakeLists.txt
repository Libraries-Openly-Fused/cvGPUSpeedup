cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 21)
set (PROJECT_VERSION_REV 0)
set (PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})

project(cvGPUSpeedup VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA 
DESCRIPTION "Make OpenCV-CUDA code run faster on the GPU" HOMEPAGE_URL "https://github.com/morousg/cvGPUSpeedup" )

include (cmake/cmake_init.cmake)
include (cmake/doxygen.cmake)
include (cmake/targets/virtualfolders.cmake)
include (cmake/deploy/deploy_dependencies.cmake)
include (cmake/archflags.cmake)
include (cmake/libs/cuda/archs.cmake)
include (cmake/generators/version_header.cmake)
include (cmake/libs/fkl.cmake)
include (cmake/libs/cuda/cuda.cmake)

include (cmake/cuda_init.cmake)
include (cmake/libs/opencv.cmake)
include (cmake/discover_tests.cmake)

option (BUILD_UTEST "build standard unit tests" ON)
option (ENABLE_BENCHMARK "build benchmarking unit tests" OFF)

add_subdirectory(include)


if (${BUILD_UTEST})
    enable_testing()  
    add_subdirectory(tests)    
endif()


if (${ENABLE_BENCHMARK})
    enable_testing()
    add_subdirectory(benchmarks)
endif()
