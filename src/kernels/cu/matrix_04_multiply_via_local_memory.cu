#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=k x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // TODO
    constexpr unsigned int TILE_K = GROUP_SIZE_X;

    __shared__ float tile_a[GROUP_SIZE_Y][TILE_K];
    __shared__ float tile_b[TILE_K][GROUP_SIZE_X];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int x = blockIdx.x * blockDim.x + tx;
    const unsigned int y = blockIdx.y * blockDim.y + ty;

    float sum = 0.0f;

    for (size_t offset = 0; offset < k; offset += TILE_K) {
        // Загружаем плитку A по строкам. За границами — нули.
        const size_t a_col = offset + tx;
        tile_a[ty][tx] = (y < h && a_col < k)
            ? a[static_cast<size_t>(y) * k + a_col]
            : 0.0f;

        // Загружаем плитку B по строкам.
        // Цикл позволяет использовать и прямоугольный блок.
        for (unsigned int row = ty; row < TILE_K; row += GROUP_SIZE_Y) {
            const size_t b_row = offset + row;
            tile_b[row][tx] = (b_row < k && x < w)
                ? b[b_row * w + x]
                : 0.0f;
        }

        // Ждём, пока весь блок загрузит обе плитки.
        __syncthreads();

        for (unsigned int i = 0; i < TILE_K; ++i) {
            sum += tile_a[ty][i] * tile_b[i][tx];
        }

        // Не перезаписываем плитки, пока другие потоки их читают.
        __syncthreads();
    }

    if (x < w && y < h) {
        c[static_cast<size_t>(y) * w + x] = sum;
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
