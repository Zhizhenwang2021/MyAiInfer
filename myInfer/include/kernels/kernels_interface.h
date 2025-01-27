//
// Created by lx186 on 2024/12/1.
//

#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H
#include <base/cuda_config.h>
#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);
typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);
typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, void* stream);
typedef void (*MatmulKernelQuant)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale, const CudaConfig* config);
    typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, float scale, const CudaConfig* config);

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

}


#endif //KERNELS_INTERFACE_H
