#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cassert>
#include <time.h>
// #include <unistd.h>
#include <algorithm>
#include <chrono>
#include <math.h>
#include <thread>
#include <functional>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "common/logger.h"

#include <opencv2/opencv.hpp>

#include "nvtx3/nvToolsExt.h"
#include "common/NvInferRuntimeCommon.h"

#include "NvDecoder/NvDecoder.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegStreamer.h"
#include "Utils/FFmpegDemuxer.h"

#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <map>
#include <math.h>
// #include "cuda_quicksort.h"

struct Job
{
    std::shared_ptr<std::promise<int>> pro;
    // std::string input;
    int device_idx;
};

const std::vector<std::string> cocolabels = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

class Yolov5TRT{

public:

    Yolov5TRT(int device_id);
    ~Yolov5TRT();

    void init(int height,int width);

    void inference_cv(cudaStream_t &stream);
    //void inference_cv();
    void inference(
        // cv::Mat &img,
        uint8_t *image_d,
        cudaStream_t &stream);



    //parameter
    int device_id;
    cudaStream_t stream;
    cudaDeviceProp prop;

    std::string engine_file;

    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *execution_context;

    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int input_numel;
    float* input_data_host;
    float* input_data_device;
    nvinfer1::Dims output_dims;
    nvinfer1::Dims input_dims;
    int output_numbox;
    int output_numprob;
    int num_classes;
    int output_numel;
    float* output_data_host;
    float* output_data_device;
    float* bbox_device;
    float* bbox_host;

    cv::Mat image;
    cv::Mat input_image;
    cv::VideoCapture cap;
    std::string url;

    // float i2d[6], d2i[6];
    float *i2d, *d2i;
    cv::Mat m2x3_i2d;
    cv::Mat m2x3_d2i;

    float *d2i_d;

    float* pRect;
    int p_num;
    const int maxSize_rect = 1000;
    int class_num;
    float* pColors;

    // uint8_t *render_surface;

};