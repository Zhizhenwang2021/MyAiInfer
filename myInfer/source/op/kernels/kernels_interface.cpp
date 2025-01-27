//
// Created by lx186 on 2024/12/1.
//

#include "kernels/kernels_interface.h"
#include <base/base.h>
#include "kernels/cpu/add_kernel.h"
#include "kernels/cuda/add_kernel.cuh"
#include "kernels/cpu/emb_kernel.h"
#include "kernels/cuda/emb_kernel.cuh"
#include "kernels/cpu/rmsnorm_kernel.h"
#include "kernels/cuda/rmsnorm_kernel.cuh"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/cuda/matmul_kernel.cuh"

namespace kernel {
    AddKernel get_add_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return add_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return add_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a add kernel.";
            return nullptr;
        }
    }

    EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return emb_kernel_normal;
        }else if (device_type == base::DeviceType::kDeviceCUDA) {
            return emb_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get an embedding kernel.";
            return nullptr;
        }
    }

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return rmsnorm_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return rmsnorm_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get an rmsnorm kernel.";
            return nullptr;
        }
    }

    MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return matmul_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return matmul_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get an matmul kernel.";
            return nullptr;
        }
    }

    MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return matmul_kernel_cu_qint8;
        } else {
            LOG(FATAL) << "Unknown device type for get an matmul kernel.";
            return nullptr;
        }
    }




}