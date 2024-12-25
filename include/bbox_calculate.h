#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>


void bbox_calculate(
    float *output_data_device,float *bbox_device,float *d2i_device,
    int output_numbox,int output_numprob,int num_classes,
    float confidence_threshold,cudaStream_t stream
);
void bbox_calculate_v8(
    float *output_data_device,float *bbox_device,float *d2i_device,
    int output_numbox,int output_numprob,int num_classes,
    float confidence_threshold,cudaStream_t stream
);