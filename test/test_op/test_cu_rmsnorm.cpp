//
// Created by lx186 on 2024/12/26.
//
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

TEST(test_rmsnorm_cu, rmsnorm_nostream) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 15;

    tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < size; ++i) {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
    }
    tensor::Tensor in_cu = in_cpu.clone();
    tensor::Tensor wei_cu = wei_cpu.clone();
    tensor::Tensor out_cu = out_cpu.clone();
    in_cu.to_cuda(nullptr);
    wei_cu.to_cuda(nullptr);
    out_cu.to_cuda(nullptr);

    kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu,
                                                            nullptr);
    out_cu.to_cpu();

    kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

    for (int i = 0; i < size; ++i) {
        ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    }
}