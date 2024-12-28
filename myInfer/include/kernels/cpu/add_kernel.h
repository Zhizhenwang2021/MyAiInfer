//
// Created by lx186 on 2024/12/1.
//

#ifndef MYINFER_INCLUDE_OP_KERNEL_ADD_H_
#define MYINFER_INCLUDE_OP_KERNEL_ADD_H_
#include "tensor/tensor.h"


namespace kernel {
    void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output, void* stream = nullptr);
}

#endif //MYINFER_INCLUDE_OP_KERNEL_ADD_H_
