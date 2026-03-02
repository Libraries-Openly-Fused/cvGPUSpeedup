if(DEFINED OpenCV_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${OpenCV_DIR}")
elseif(DEFINED OPENCV_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${OPENCV_DIR}")
endif()
find_package(OpenCV ${OPENCV_VERSION} CONFIG REQUIRED)

function(add_opencv_to_target TARGET_NAME COMPONENTS)
    set(EXPORTED_TARGETS ${COMPONENTS})
	
    list(TRANSFORM EXPORTED_TARGETS PREPEND "opencv_")
    
	target_link_libraries(${TARGET_NAME} PRIVATE ${EXPORTED_TARGETS})
    if(WIN32)
        deploy_exported_target_dependencies(${TARGET_NAME} ${EXPORTED_TARGETS})
    endif()
    # cuda libraries  implicit dependencies for cuda modules
    set(NPP_COMPONENTS nppc nppial nppicc nppig nppidei nppitc nppist nppc nppif nppim npps)
    if(NOT UNIX)
        deploy_cuda_dependencies(${TARGET_NAME} "${NPP_COMPONENTS}")
        # cudaarithm
        deploy_cuda_dependencies(${TARGET_NAME} "cufft;cublas;cublasLt")

    endif()

endfunction()
