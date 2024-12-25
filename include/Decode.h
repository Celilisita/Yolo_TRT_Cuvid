#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/ColorSpace.h"
#include "Common/AppDecUtils.h"

#include "nvtx3/nvToolsExt.h"
// #include "FramePresenterGLUT.h"


#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include "FramePresenter.h"
// #include "FramePresenterGLUT.h"

#include "FramePresenterD3D.h"
// #include "FramePresenterD3D9.h"
#include "FramePresenterD3D11.h"

// #if !defined (_WIN32)
// #include "FramePresenterGLX.h"


// #endif

#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

enum OutputFormat
{
    native = 0, bgrp, rgbp, bgra, rgba, bgra64, rgba64
};


class DecodeGL{

public:

    DecodeGL();
    ~DecodeGL();

    void set_url(std::string stream_url,CUcontext &cuContext);

    void GetImage(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight);

    std::string GetSupportedFormats();

    void initFrame(int rows,int cols);
    void initContext(CUcontext &cuContext);

    int decodeFrame(
        uint8_t *bgr_frame,
        uint8_t *bgra_frame,
        CUdeviceptr &pTmpImage,
        cudaStream_t &stream,
        int idx,int idj);


    std::string url;
    // enum OutputFormat={native = 0, bgrp, rgbp, bgra, rgba, bgra64, rgba64};
    
    // int iGpu;
    bool bReturn;
    // CUdeviceptr pTmpImage;
    // uint8_t *tmp_bgr;

    // cv::Mat cv_img_bgr;
    int height;
    int width;
    int channel;
    int size_frame_bgra;
    int size_frame_bgr;

    // CUcontext cuContext;

    FFmpegDemuxer *demuxer;
    NvDecoder *dec;

    // FramePresenterGLX *gInstance;
    // FramePresenterGLUT *gInstance;

    int nWidth;
    int nHeight;
    int nFrameSize;
    int *anSize;
    std::unique_ptr<uint8_t[]> pImage;
    int nVideoBytes;
    int nFrameReturned;
    int nFrame;
    int iMatrix;

    int nPitch;

    uint8_t *pVideo;
    uint8_t *pFrame;
    uint8_t *tmp;

    int m_id;
    unsigned long long attr;
    bool is_exist=false;
    bool start_flag=false;

    OutputFormat eOutputFormat = bgra;

};

// std::map<int,unsigned long long> attach_ptr;

// uint8_t* really_ptr;