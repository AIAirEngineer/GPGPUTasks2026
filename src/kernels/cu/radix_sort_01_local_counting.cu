#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    // TODO
    const unsigned int n = a1;
    const unsigned int bit = a2;

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + tid;

    // одномерный запуск не более 1024 потока в блоке
    __shared__ unsigned int local_prefix[1024];

    // За пределами входного массива вклад в сумму равен нулю
    unsigned int is_zero = 0;
    if (i < n) {
        const unsigned int current_bit = (buffer1[i] >> bit) & 1u;
        is_zero = 1u - current_bit;
    }

    local_prefix[tid] = is_zero;
    __syncthreads();

    // scan флаго is_zero внутри каждого блока
        // Inclusive scan флагов is_zero внутри каждого блока.
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        unsigned int add = 0;
        if (tid >= offset) {
            add = local_prefix[tid - offset];
        }

        // Все потоки должны закончить чтение старого состояния.
        __syncthreads();

        local_prefix[tid] += add;

        // Все потоки должны закончить запись нового состояния.
        __syncthreads();
    }

    // Сохраняем готовые локальные префиксы в выходной массив.
    if (i < n) {
        buffer2[i] = local_prefix[tid];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
