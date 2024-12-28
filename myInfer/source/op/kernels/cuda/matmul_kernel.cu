//
// Created by lx186 on 2024/12/13.
//
#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <kernels/kernels_interface.h>
#include <kernels/cuda/matmul_kernel.cuh>

namespace kernel {
    template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
    __global__ void matmul_kernel_cu_fp32(const float* input, const float* weight,
        float* output, int M, int K) {

        __shared__ float sdata[THREAD_PER_BLOCK];
        unsigned int tid = threadIdx.x;

        int start_row = blockIdx.x * ROW_PER_BLOCK;
        int end_row = start_row + ROW_PER_BLOCK;
        if (start_row >= K) {
            return;
        }

        constexpr int pack_size = 4;
        const int pack_num = M / pack_size;
        const int pack_off = pack_size * pack_num;

    #pragma unroll
        for(int p = start_row; p < end_row; ++p) {
            sdata[tid] = 0;
            int row_offset = p * M;
            float4* input_float4_ptr = (float4*)input;
            float4* weight_float4_ptr = (float4*)(weight + row_offset);

    #pragma unroll
        for(int i = tid; i < pack_num; i += blockDim.x) {
            float4 input_float4 = *(input_float4_ptr + i);
            float4 weight_float4 = *(weight_float4_ptr + i);
            float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;

            sdata[tid] += part_sum;
        }
            for (int i = pack_off + tid; i < M; i += blockDim.x) {
                sdata[tid] += input[i] * weight[row_offset + i];
            }

            __syncthreads();

            using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
            __shared__ typename BlockReduce::TempStorage temp;
            float part_sum = BlockReduce(temp).Sum(sdata[tid]);
            __syncthreads();

            if (tid == 0) {
                output[p] = part_sum;
            }
            __syncthreads();

        }
    }
}