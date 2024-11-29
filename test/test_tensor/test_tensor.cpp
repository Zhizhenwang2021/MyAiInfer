//
// Created by lx186 on 2024/11/27.
//
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"

TEST(test_tensor, to_cpu) {
    using namespace base;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1_cu.is_empty(), false);
    set_value_cu(t1_cu.ptr<float>(), 32 * 32);

    t1_cu.to_cpu();
    ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
    float* cpu_ptr = t1_cu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(*(cpu_ptr + i), 1.f);
    }
}