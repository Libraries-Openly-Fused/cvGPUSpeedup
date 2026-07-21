file(
    GLOB_RECURSE
    CUDA_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
)
add_library(headers INTERFACE ${CUDA_SOURCES})
# Set virtual folders for MSVC
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${CUDA_SOURCES})
#target_include_directories(include INTERFACE .)
set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
