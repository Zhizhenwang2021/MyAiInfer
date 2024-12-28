//
// Created by lx186 on 2024/12/8.
//

#ifndef MYINFER_INCLUDE_OP_KERNEL_EMB_H
#define MYINFER_INCLUDE_OP_KERNEL_EMB_H
#include "tensor/tensor.h"

namespace kernel{
    void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t vocab_size,
                           void* stream = nullptr);
}

#endif //MYINFER_INCLUDE_OP_KERNEL_EMB_H
