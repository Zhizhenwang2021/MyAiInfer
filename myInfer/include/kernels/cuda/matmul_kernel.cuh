//
// Created by lx186 on 2024/12/13.
//

#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH
#include "kernels/kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
    void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);

    void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t group_size,
                                const tensor::Tensor& scale, const CudaConfig* config = nullptr);

}

#endif //MATMUL_KERNEL_CUH
