
cmake_minimum_required(VERSION 3.17)


project(perf_test)



add_executable(perf_test
main.cpp
runner.h
runner.cpp
test_case.h
test_case.cpp
log.h
log.cpp
profiler.h
profiler.cpp
cuda_random.h
cuda_random.cu
composed_test_case.h
devices.h
problems.h
compute_kernels.h
loop.h
loop.cu
type_list.h
types_helper.h
struct_helper.h
class_reflection.h
macro_loop.h
cuda_device.h
launch_config.h
reduction_sum_cpu.h
reduction_sum_cpu.cpp
reduction_sum_cuda.h
reduction_sum_cuda.cpp
serial_quick_sort.h
serial_quick_sort.cpp
tbb_quick_sort.h
tbb_quick_sort.cpp
)


target_include_directories(perf_test PUBLIC external/spdlog/include)

target_compile_features(perf_test PRIVATE cxx_std_20)

include(FindCUDAToolkit)
if(CUDAToolkit_FOUND)
	message("CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
	message("CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
	message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
	message("CUDAToolkit_TARGET_DIR: ${CUDAToolkit_TARGET_DIR}")
	message("CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")

	enable_language(CUDA)
endif()


target_link_libraries(perf_test PUBLIC curand)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	message("omp found!")
    target_link_libraries(perf_test PUBLIC OpenMP::OpenMP_CXX)
else()
	message("omp not found!")
endif()

find_package(TBB CONFIG REQUIRED)
target_link_libraries(perf_test PRIVATE TBB::tbb)

add_custom_command(TARGET perf_test POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
	$<TARGET_FILE_DIR:TBB::tbb>/../bin/tbb$<$<CONFIG:Debug>:_debug>.dll $<TARGET_FILE_DIR:perf_test>) 
