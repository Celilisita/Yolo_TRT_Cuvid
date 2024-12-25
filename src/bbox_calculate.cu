#include "bbox_calculate.h"


__device__ int _max_element_order(float *pclass,int num_classes){
    int max_idx=0;
    float max_value=-1048576.0f;
    for(int i=0;i<num_classes;i++){
        if(pclass[i]>max_value){
            max_value=pclass[i];
            max_idx=i;
        }
    }

    return max_idx;
}

__device__ int sum_shared_bbox(int num_flag[]){
    int sum=0;
    for(int i=0;i<32;i++){
        if(num_flag[i]>0){
            sum++;
        }
    }
    return sum;
}

__global__ void bbox_cal_kernel(float *output_dst,float *bbox,float *d2i,int numbox,int numprob,int num_classes,float confidence_threshold){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;

    // __shared__ int num_flag[32];
    // num_flag[threadIdx.x]=0;

    if(idx>=numbox*numprob){
        return;
    }
    
    int ptr_idx=idx*numprob;

    float objness=output_dst[ptr_idx+4];
    if(objness<confidence_threshold){
        return;
    }

    int label=_max_element_order(output_dst+ptr_idx+5,num_classes);
    float prob=output_dst[ptr_idx+5+label];
    float confidence=prob*objness;
    if(confidence<confidence_threshold){
        return;
    }

    float cx=output_dst[ptr_idx];
    float cy=output_dst[ptr_idx+1];
    float width=output_dst[ptr_idx+2];
    float height=output_dst[ptr_idx+3];

    float left=cx-width*0.5;
    float top=cy-height*0.5;
    float right=cx+width*0.5;
    float bottom=cy+height*0.5;

    left=d2i[0]*left+d2i[2];
    right=d2i[0]*right+d2i[2];
    top=d2i[0]*top+d2i[5];
    bottom=d2i[0]*bottom+d2i[5];

    // num_flag[threadIdx.x]=1;

    // __syncthreads();

    // int sum_thread=sum_shared_bbox(num_flag);

    // int num_bbox=(int)bbox[0];
    // if(num_bbox>=1000){
    //     return;
    // }

    bbox[idx*7+1]=left;
    bbox[idx*7+2]=top;
    bbox[idx*7+3]=right;
    bbox[idx*7+4]=bottom;
    bbox[idx*7+5]=(float)label;
    bbox[idx*7+6]=confidence;
    bbox[idx*7+7]=1.0f;
    atomicAdd(&bbox[0],1.0f);
    // atomicAdd(&bbox[0],sum_thread);
    // atomicExch(&bbox[0],sum_thread+num_bbox);
    // num_bbox++;

}
__global__ void bbox_cal_v8_kernel(float *output_dst,float *bbox,float *d2i,int numbox,int numprob,int num_classes,float confidence_threshold){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;

    // __shared__ int num_flag[32];
    // num_flag[threadIdx.x]=0;

    if(idx>=numbox*numprob){
        return;
    }
    
    int ptr_idx=idx*numprob;

    // float objness=output_dst[ptr_idx+4];
    // if(objness<confidence_threshold){
    //     return;
    // }

    int label=_max_element_order(output_dst+ptr_idx+4,num_classes);
    float prob=output_dst[ptr_idx+4+label];
    float confidence=prob;
    if(confidence<confidence_threshold){
        return;
    }

    float cx=output_dst[ptr_idx];
    float cy=output_dst[ptr_idx+1];
    float width=output_dst[ptr_idx+2];
    float height=output_dst[ptr_idx+3];

    float left=cx-width*0.5;
    float top=cy-height*0.5;
    float right=cx+width*0.5;
    float bottom=cy+height*0.5;

    left=d2i[0]*left+d2i[2];
    right=d2i[0]*right+d2i[2];
    top=d2i[0]*top+d2i[5];
    bottom=d2i[0]*bottom+d2i[5];

    // num_flag[threadIdx.x]=1;

    // __syncthreads();

    // int sum_thread=sum_shared_bbox(num_flag);

    // int num_bbox=(int)bbox[0];
    // if(num_bbox>=1000){
    //     return;
    // }

    bbox[idx*7+1]=left;
    bbox[idx*7+2]=top;
    bbox[idx*7+3]=right;
    bbox[idx*7+4]=bottom;
    bbox[idx*7+5]=(float)label;
    bbox[idx*7+6]=confidence;
    bbox[idx*7+7]=1.0f;
    atomicAdd(&bbox[0],1.0f);
    // atomicAdd(&bbox[0],sum_thread);
    // atomicExch(&bbox[0],sum_thread+num_bbox);
    // num_bbox++;

}

void bbox_calculate(
    float *output_data_device,float *bbox_device,float *d2i_d,
    int output_numbox,int output_numprob,int num_classes,
    float confidence_threshold,cudaStream_t stream
){
    //gpu kernel
    dim3 threadSize(32);
    dim3 blockSize((output_numbox+31)/32);
    bbox_cal_kernel<<<blockSize,threadSize,0,stream>>>(
        output_data_device,bbox_device,d2i_d,
        output_numbox,output_numprob,num_classes,
        confidence_threshold);
}

void bbox_calculate_v8(
    float *output_data_device,float *bbox_device,float *d2i_d,
    int output_numbox,int output_numprob,int num_classes,
    float confidence_threshold,cudaStream_t stream
){
    //gpu kernel
    dim3 threadSize(32);
    dim3 blockSize((output_numbox+31)/32);
    bbox_cal_v8_kernel<<<blockSize,threadSize,0,stream>>>(
        output_data_device,bbox_device,d2i_d,
        output_numbox,output_numprob,num_classes,
        confidence_threshold);
}