#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
    const unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    // TODO
    const unsigned int n = a1;
    const unsigned int bit = a2;

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {return;}

    // buffer1 — числа текущего прохода
    // buffer2 — глобальный inclusive scan флагов is_zero для этих чисел
    // buffer3 — отдельный выходной массив; не перекрывается с входами
    const unsigned int value = buffer1[i];
    const unsigned int current_bit = (value >> bit) & 1u;

    const unsigned int zeros_prefix = buffer2[i];
    const unsigned int total_zeros = buffer2[n - 1u];

    unsigned int dst;
    if (current_bit == 0u) {
        // Номер среди элементов с нулевым битом, начиная с нуля
        dst = zeros_prefix - 1u;
    } else {
        // Пропускаем элементы с нулевым битом и предыдущие с единичным
        dst = total_zeros + (i - zeros_prefix);
    }

    buffer3[dst] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
