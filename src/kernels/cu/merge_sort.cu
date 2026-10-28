#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO
    if (i >= n) {return;}

    // find two sorted arrays for position i
    const int pair_size = 2 * sorted_k;
    const int pair_begin = (i / pair_size) * pair_size;
    const int left_size = min(sorted_k, n - pair_begin);
    const int right_size = min(sorted_k, n - pair_begin - left_size);

    const unsigned int* left = input_data + pair_begin;
    const unsigned int* right = left + left_size;

    // result position inside these two arrays
    const int k = i - pair_begin;

    // search how much elements from left enter in firstly k result elements
    int low = max(0, k - right_size);
    int high = min(k, left_size);

    while (low <= high) {
        const int a = low + (high - low) / 2;
        const int b = k - a;

        if (a > 0 && b < right_size && left[a - 1] > right[b]) {
            // if take too much elements from left
            high = a - 1;
        } else if (b > 0 && a < left_size && right[b - 1] >= left[a]) {
            // take мало elements from left
            // if equal left elem must be firstly thm right
            low = a + 1;
        } else {
            // we found the edge. Choose next ele,after first k
            if (a < left_size && (b == right_size || left[a] <= right[b])) {
                output_data[i] = left[a];
            } else {
                output_data[i] = right[b];
            }
            return;
        }
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
