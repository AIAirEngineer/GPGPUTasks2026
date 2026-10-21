#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1)
{
    // TODO
    const unsigned int n = a1;
    const unsigned int block_size = blockDim.x;
    const unsigned int tid = threadIdx.x;

    // Один блок обрабатывает одну группу из block_size входных эл-ов
    const unsigned int group = blockIdx.x * block_size + tid;
    const unsigned int groups_count = n / block_size + (n % block_size != 0);
    // Одномерный запуск: 2 <= block_size <= 1024
    // Размер блока должен совпадать с размером групп входного Scan
    // buffer1 и buffer2 — отдельные, неперекрывающиеся буферы
    __shared__ unsigned int group_prefix[1024];

    unsigned int group_sum = 0;
    if (group < groups_count) {
        const unsigned int start = group * block_size;

        // У последней группы может быть меньше block_size элементов
        unsigned int count = n - start;
        if (count > block_size) {
            count = block_size;
        }

        const unsigned int last = start + count - 1;
        group_sum = buffer1[last];
    }
    group_prefix[tid] = group_sum;
    __syncthreads();

    // Inclusive scan итогов групп внутри текущего CUDA-блока
    for (unsigned int offset = 1; offset < block_size; offset *= 2) {
        unsigned int add = 0;
        if (tid >= offset) {
            add = group_prefix[tid - offset];
        }
        // Завершаем чтение старого состояния до начала обновлений
        __syncthreads();

        group_prefix[tid] += add;

        // Завершаем обновления до следующего шага Scan
        __syncthreads();

    }

    if (group < groups_count) {
        buffer2[group] = group_prefix[tid];
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
