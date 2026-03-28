 function(add_vertical_fusion_kernels TARGET_NAME FOLDER_FROM)
    file(
        GLOB_RECURSE
        VF_KERNEL_SOURCES
        CONFIGURE_DEPENDS
        "${FOLDER_FROM}/*"
    )
    target_sources(${TARGET_NAME} PRIVATE ${VF_KERNEL_SOURCES})
endfunction()# --- Configuration ---


function (generate_vf_kernels NUM_EXPERIMENTS_AS_INT GENERATED_DIR GENERATED_CU_FILES_LIST OUT_LAUNCHER_H_FILE OPTYPE)
    foreach(IDX RANGE 1 ${NUM_EXPERIMENTS_AS_INT})
        set(N ${IDX}) # Set the variable 'N' that @N@ will be replaced with

        set(CURRENT_MUL_H "${GENERATED_DIR}/kernel${N}.h")
        set(CURRENT_MUL_CU "${GENERATED_DIR}/kernel${N}.cu")

        # Generate mulN.h
        configure_file("${IN_KERNEL_H}" "${CURRENT_MUL_H}" @ONLY)

        # Generate mulN.cu
        configure_file("${IN_KERNEL_CU}" "${CURRENT_MUL_CU}" @ONLY)

        list(APPEND GENERATED_CU_FILES_LIST ${CURRENT_MUL_CU})

        # Append to strings for the main launcher header
        string(APPEND LAUNCHER_INCLUDES_BLOCK "#include \"kernel${N}.h\"\n")
        string(APPEND LAUNCHER_DISPATCH_BLOCK  "DISPATCH_INSTANCE(${N})\n")
    endforeach()

    
    # --- Generate the Main Dispatcher Header (mul_launcher.h) ---
    set(GENERATED_INCLUDES ${LAUNCHER_INCLUDES_BLOCK})
    set(GENERATED_DISPATCH_CALLS ${LAUNCHER_DISPATCH_BLOCK})
    set(NUM_EXPERIMENTS_TOTAL ${NUM_EXPERIMENTS_AS_INT}) # For static_assert in template

    set(OUT_LAUNCHER_H_FILE1 "${GENERATED_DIR}/launcher.h")
    configure_file("${IN_LAUNCHER_H}" "${OUT_LAUNCHER_H_FILE1}" @ONLY)
    set(OUT_LAUNCHER_H_FILE "${OUT_LAUNCHER_H_FILE1}" PARENT_SCOPE)
    
endfunction()

function (add_vertical_fusion_benchmark TARGET_NAME GENERATED_DIR OPTYPE)
    # --- File Generation Loop ---
    set(GENERATED_CU_FILES_LIST "")    # To collect all mulN.cu files for the executable
    set(LAUNCHER_INCLUDES_BLOCK "")    # To build the #include block for mul_launcher.h
    set(LAUNCHER_DISPATCH_BLOCK "") # To build the DISPATCH_TO_MUL_INSTANCE block
    set(TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../templates") # Store your .in files here
    # --- Template File Names ---
    set(IN_KERNEL_H "${TEMPLATE_DIR}/experiment_kernel.h.in")
    set(IN_KERNEL_CU "${TEMPLATE_DIR}/experiment_kernel.cu.in")
    set(IN_LAUNCHER_H "${TEMPLATE_DIR}/experiment_launcher.h.in") # Main dispatcher template

 
    generate_vf_kernels("${NUM_EXPERIMENTS_AS_INT}" "${GENERATED_DIR}" "${GENERATED_CU_FILES_LIST}" "${OUT_LAUNCHER_H_FILE}" "${OPTYPE}")         
    add_executable(${TARGET_NAME}  launcher.cu opType.cuh realBatch.h ${GENERATED_CU_FILES_LIST} "${OUT_LAUNCHER_H_FILE}"
        ${GENERATED_DIR}/launcher.h)
        
    add_vertical_fusion_kernels(${TARGET_NAME}  ${GENERATED_DIR})
    add_cuda_to_target(${TARGET_NAME} "")

    # --- Include Directories ---
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include    
        ${BENCHMARKS_INCLUDE_ROOT}           # For <benchmarks/opencv/...>
        ${GENERATED_DIR}                     # For "mulN.h", "mul_launcher.h"
        
    )
    add_fkl_to_target(${TARGET_NAME})                 
    add_opencv_to_target(${TARGET_NAME} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
    enable_intellisense(${TARGET_NAME} "verticalfusion")
    
    # To pass NUM_EXPERIMENTS_AS_INT to your C++ code (e.g., main.cu)
    # In C++: int max_exp = CPP_NUM_EXPERIMENTS;
    target_compile_definitions(${TARGET_NAME} PRIVATE CPP_NUM_EXPERIMENTS=${NUM_EXPERIMENTS_AS_INT})
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-split-compile=0 --fdevice-time-trace=${TARGET_NAME}>)
endfunction()

function (add_single_benchmark TARGET_NAME DIR_NAME)
    # --- File Generation Loop ---
  
    add_executable(${TARGET_NAME}  ${TARGET_NAME}.cu)            
    add_cuda_to_target(${TARGET_NAME} "")

    # --- Include Directories ---
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include    
        ${BENCHMARKS_INCLUDE_ROOT}           # For <benchmarks/opencv/...>        
        
    )
    add_fkl_to_target(${TARGET_NAME})                 
    add_opencv_to_target(${TARGET_NAME} "core;cudaarithm;imgproc;cudafilters;cudaimgproc;cudawarping;imgcodecs")
    enable_intellisense(${TARGET_NAME} "${DIR_NAME}")
    
endfunction()