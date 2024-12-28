//
// Created by lx186 on 2024/12/24.
//
#include <cub/block/block_reduce.cuh>
#include "kernels/cuda/rmsnorm_kernel.cuh"

#include <crt/math_functions.h>

namespace kernel {
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, const int size, float eps) {
    // const int t_n = gridDim.x * blockDim.x;
    //
    // int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    //
    // float sum = 0.0f;
    //
    // __shared__ float val[32];
    //
    // while (tid < size) {
    //     val[tid] = in[tid] * in[tid];
    //     tid +=t_n;
    // }
    // __syncthreads();
    //
    // for (int i = 0; i < size; i++) {
    //     sum += val[i];
    // }
    // const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    // for (int i = threadIdx.x; i < size; i += blockDim.x) {
    //     out[i] = wei[i] * in[i] * scale;
    // }

    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
        // printf("blockId.x:%d threadIdx.x:%d in[i]:%f sum:%f \n", blockIdx.x, threadIdx.x, in[i], sum);
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);

    if (threadIdx.x == 0) {
        shared_val = sum;
    }

    __syncthreads();
    sum = shared_val;
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for (int i = tid; i < size; i += blockDim.x) {
        out[i] = wei[i] * in[i] * scale;
    }

    // const int tid = threadIdx.x; //块中第几个线程
    //
    // constexpr int pack_size = 4; //块大小
    // const int pack_num = size / pack_size; // 需要几块
    // const int pack_off = pack_size * pack_num; //一共多少线程
    //
    // float sum = 0.0f;
    // float4* in_pack = reinterpret_cast<float4*>(in);
    // for (int i = tid; i < pack_num; i += blockDim.x) {
    //     float4 in_float4 = *(in_pack + i);
    //     sum += in_float4.x * in_float4.x;
    //     sum += in_float4.y * in_float4.y;
    //     sum += in_float4.z * in_float4.z;
    //     sum += in_float4.w * in_float4.w;
    // }
    //
    // for (int i = pack_off + tid; i < size; i += blockDim.x) {
    //     sum += in[i] * in[i];
    // }
    //
    // using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    // __shared__ typename BlockReduce::TempStorage temp;
    // __shared__ float shared_val;
    // sum = BlockReduce(temp).Sum(sum);
    // if (threadIdx.x == 0) {
    //     shared_val = sum;
    // }
    // __syncthreads();
    // sum = shared_val;
    // const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    //
    // float4* wei_pack = reinterpret_cast<float4*>(wei);
    // float4* out_pack = reinterpret_cast<float4*>(out);
    // for (int i = tid; i < pack_num; i += blockDim.x) {
    //     float4 in_float4 = *(in_pack + i);
    //     float4 wei_float4 = *(wei_pack + i);
    //     *(out_pack + i) =
    //         make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
    //                     scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    // }
    //
    // for (int i = pack_off + tid; i < size; i += blockDim.x) {
    //     out[i] = wei[i] * in[i] * scale;
    // }
}

void rmsnorm_kernel_cu(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output, void *stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
          weight.device_type() == base::DeviceType::kDeviceCUDA &&
          output.device_type() == base::DeviceType::kDeviceCUDA);

#ifdef QWEN2_SUPPORT
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif
    const int32_t size = static_cast<int32_t>(input.size());
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    constexpr int threads_num = 128;

    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<threads_num><<<128, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
        row_rmsnorm_f32<threads_num><<<128, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }

}
}
