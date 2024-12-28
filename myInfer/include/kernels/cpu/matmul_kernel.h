//
// Created by lx186 on 2024/12/13.
//

#ifndef MYINFER_INCLUDE_OP_KERNEL_MATMUL_H
#define MYINFER_INCLUDE_OP_KERNEL_MATMUL_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
    void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
        const tensor::Tensor& output, float scale = 1.f,
        const CudaConfig* config = nullptr);
}
#endif //MYINFER_INCLUDE_OP_KERNEL_MATMUL_H
