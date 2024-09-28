cmake_minimum_required(VERSION 3.16)

project(my_infer)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)


find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(myInfer/source/base/ DIR_BASE)

#add_library(infer SHARED ${DIR_BASE} myInfer/include)
add_executable(infer ${DIR_BASE})

target_include_directories(infer PUBLIC ${PROJECT_SOURCE_DIR}/myInfer/include)

#add_executable(my_app main.cpp)
#target_include_directories(my_app PRIVATE /path/to/include)