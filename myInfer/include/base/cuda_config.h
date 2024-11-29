//
// Created by lx186 on 2024/11/27.
//

#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include<cublas_v2.h>
#include<cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
    cudaStream_t stream = nullptr;
    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};
}

#endif //BLAS_HELPER_H
