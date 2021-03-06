
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
vector_add_cuda.h
vector_add_cuda.cu
vector_add_omp.h
vector_add_omp.cpp
cuda_random.h
cuda_random.cu

)


target_include_directories(perf_test PUBLIC external/spdlog/include)

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
