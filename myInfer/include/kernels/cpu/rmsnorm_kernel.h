//
// Created by lx186 on 2024/12/15.
//

#ifndef MYINFER_INCLUDE_OP_KERNEL_RMSNORM_H
#define MYINFER_INCLUDE_OP_KERNEL_RMSNORM_H
#include "tensor/tensor.h"
namespace kernel {
    void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
        const tensor::Tensor& output, void* stream = nullptr);
}


#endif //MYINFER_INCLUDE_OP_KERNEL_RMSNORM_H
