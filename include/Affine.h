#include <cuda_runtime.h>


void warp_affine_bilinear(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value,cudaStream_t stream
);

void resize_nearest_gpu(uint8_t *src,uint8_t *dst,int src_width,int src_height,
    int dst_width,int dst_height,cudaStream_t stream);
