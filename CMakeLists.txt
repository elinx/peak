cmake_minimum_required(VERSION 3.20)
project(peak CXX)

add_executable(mkl mkl_dgemm.cpp)
target_link_libraries(mkl "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core")

add_executable(naive-ijk naive_ijk.cpp)
add_executable(manual manual_optimize_dgemm.cpp)
add_executable(openblas openblas_dgemm.cpp)
target_link_libraries(openblas "-lopenblas")
