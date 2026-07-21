set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h;")
if (WIN32)
    list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/manifest.xml")
endif()
 
function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.cu"
    )
     
    foreach(cuda_source ${CUDA_SOURCES})
        get_filename_component(cuda_target ${cuda_source} NAME_WE)           
        add_executable(${cuda_target} ${cuda_source} )
        add_fkl_to_target(${cuda_target})
        target_sources(${cuda_target} PRIVATE ${LAUNCH_SOURCES})            
     
        if(${ENABLE_BENCHMARK})
            target_compile_definitions(${cuda_target} PRIVATE ENABLE_BENCHMARK)
        endif()
          
        cmake_path(SET path2 "${DIR}")
        cmake_path(GET path2 FILENAME DIR_NAME)       
        set_property(TARGET ${cuda_target} PROPERTY FOLDER tests/${DIR_NAME})
        add_cuda_to_target(${cuda_target} "")
            
        if(${ENABLE_DEBUG})
            add_cuda_debug_support_to_target(${cuda_target})
        endif()

        if(${ENABLE_NVTX})
            add_nvtx_support_to_target(${cuda_target})
        endif()

        set_target_properties(${cuda_target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
        target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}/")      
        target_include_directories(${cuda_target} PRIVATE "${CMAKE_SOURCE_DIR}/include")          
        #target_link_libraries(${cuda_target} PRIVATE ${PROJECT_NAME})        
        add_opencv_to_target(${cuda_target} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
        set_target_cuda_arch_flags(${cuda_target})
        add_test(NAME  ${cuda_target} COMMAND ${cuda_target})
        add_optimization_flags(${cuda_target})
    	string(FIND ${cuda_source} "npp" is_npp)    	
		 
		if (${is_npp} GREATER -1)		    
			target_link_libraries(${cuda_target} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig) 								              
		endif()
		
    endforeach()
endfunction()