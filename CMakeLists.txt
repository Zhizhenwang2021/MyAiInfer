cmake_minimum_required(VERSION 3.0.0)

project(MyAiInfer)
include_directories(${PROJECT_SOURCE_DIR} dirs)
set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_LIST)

link_directories(${PROJECT_SOURCE_DIR}/lib)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_executable(app ${SRC_LIST})

#link_libraries(calc)
#target_link_directories(${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(app test)
#add_library(test SHARED ${SRC_LIST})
