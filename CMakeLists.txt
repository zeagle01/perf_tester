
cmake_minimum_required(VERSION 3.15)


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
)


target_include_directories(perf_test PUBLIC external/spdlog/include)