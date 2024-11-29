cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")

project(my_infer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)


aux_source_directory(myInfer/source/base/ DIR_BASE)
aux_source_directory(myInfer/source/tensor/ DIR_TENSOR)
aux_source_directory(myInfer/source/op/ DIR_OP)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(infer SHARED ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP})
#add_executable(infer ${DIR_BASE})

target_link_libraries(infer sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(infer PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(infer PUBLIC ${PROJECT_SOURCE_DIR}/myInfer/include)
target_include_directories(infer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(infer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)

#add_executable(my_app main.cpp)
#target_include_directories(my_app PRIVATE /path/to/include)