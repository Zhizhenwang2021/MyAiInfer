include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/test_op DIR_TEST_OP)

add_executable(test_infer ${DIR_TEST} ${DIR_TEST_TENSOR} ${DIR_TEST_OP})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_infer ${link_ext_lib})
target_include_directories(test_infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_infer PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_infer PUBLIC ../myInfer/include)
target_link_directories(test_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_infer infer)

set_target_properties(test_infer PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)