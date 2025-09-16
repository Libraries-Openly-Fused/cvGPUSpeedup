if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")
endif()
 
# cuda version and compiler detection
include(cmake/cuda_init.cmake)

if(MSVC)
    set(OUT_DIR "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
    
else()
    set(OUT_DIR "${CMAKE_BINARY_DIR}/bin/")
endif()
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)

# If CMake does not have a mapping for RelWithDebInfo in imported targets it will map those configuration to the first
# valid configuration in CMAKE_CONFIGURATION_TYPES. By default this is the debug configuration which is wrong. See:
# https://gitlab.kitware.com/cmake/cmake/-/issues/20319
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO "RelWithDebInfo;Release;")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

set (TEMPLATE_DEPTH "default" CACHE STRING  "template depth")


cmake_policy(SET CMP0111 NEW) # ensure all targets provide shared libs location


