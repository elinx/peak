cmake_minimum_required(VERSION 3.20)
project(peak CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
add_definitions("-mavx2 -march=native")

add_executable(mkl mkl_dgemm.cpp)
target_link_libraries(mkl "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core")

add_executable(naive-ijk naive_ijk.cpp)
add_executable(manual manual_optimize_dgemm.cpp)
add_executable(openblas openblas_dgemm.cpp)
target_link_libraries(openblas "-lopenblas")

find_package(Halide)
add_executable(halide_reference halide_reference.cpp)
target_link_libraries(halide_reference Halide::Halide "-lopenblas")
