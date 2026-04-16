
include(cmake/intellisense.cmake)
include(cmake/benchmark.cmake)
 
set(NUM_EXPERIMENTS 5 CACHE STRING "Number of experiments to generate (e.g., 5)")
option (NUM_EXPERIMENTS "Number of experiments to generate (e.g., 5)" ${NUM_EXPERIMENTS})
math(EXPR NUM_EXPERIMENTS_AS_INT "${NUM_EXPERIMENTS}") # Convert to integer for loop
if(NUM_EXPERIMENTS_AS_INT LESS_EQUAL 0)
    message(FATAL_ERROR "NUM_EXPERIMENTS must be a positive integer. Value: ${NUM_EXPERIMENTS}")
endif()
# Source directory for <benchmarks/...> includes
# Adjust this if your 'benchmarks' directory is elsewhere (e.g., inside a 'src' folder)
set(BENCHMARKS_INCLUDE_ROOT "${CMAKE_SOURCE_DIR}")


add_single_benchmark("benchmark_CPU_OpenCV_vs_cvGS" "")
add_single_benchmark("benchmark_CPUandGPU_cvGS_vs_fk" "")
add_single_benchmark("benchmark_image_resolution_MAD_loop" "")

add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/add)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/div)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/sub)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/mul)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/mul1)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/mul_add)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/mul_add1)
add_subdirectory(verticalfusion/vertical_fusion_kernel_instances/compile_times)

