//
// Created by lx186 on 2024/12/8.
//

#ifndef EMB_KERNEL_CUH
#define EMB_KERNEL_CUH
#include "tensor//tensor.h"
namespace kernel {
    void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);
}
#endif //EMB_KERNEL_CUH
