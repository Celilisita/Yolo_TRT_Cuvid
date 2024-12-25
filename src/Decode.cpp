
// #include <cuda.h>
// #include <iostream>
// #include <algorithm>
// #include <memory>
// #include "NvDecoder/NvDecoder.h"
// #include "../Utils/FFmpegDemuxer.h"
// #include "../Utils/ColorSpace.h"
// #include "../Common/AppDecUtils.h"

// #include "opencv2/opencv.hpp"
// #include "opencv2/highgui.hpp"

#include "Decode.h"
#include "Affine.h"

// simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


DecodeGL::DecodeGL(){
    url="";
    // OutputFormat={native = 0, bgrp, rgbp, bgra, rgba, bgra64, rgba64};
    // iGpu=0;
    bReturn=1;
    // pTmpImage=0;

}

DecodeGL::~DecodeGL(){

}

void DecodeGL::set_url(std::string stream_url,CUcontext &cuContext){
    this->url=stream_url;
    initContext(cuContext);
}

void DecodeGL::initContext(CUcontext &cuContext){
    // ck(cuInit(0));
    // int nGpu=0;
    // ck(cuDeviceGetCount(&nGpu));
    // if (iGpu < 0 || iGpu >= nGpu)
    // {
    //     std::ostringstream err;
    //     err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
    //     throw std::invalid_argument(err.str());
    // }

    // cuContext=NULL;
    // createCudaContext(&cuContext,iGpu,0);

    if(url.size()>0){
        demuxer=new FFmpegDemuxer(url.data());
    }else{
        std::ostringstream err;
        err << "Input url error";
        throw std::invalid_argument(err.str());
    }

    // CUcontext cuContext1 = NULL;
    // createCudaContext(&cuContext1, 1, CU_CTX_SCHED_BLOCKING_SYNC);
    // dec=new NvDecoder(cuContext1, true, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));

    dec=new NvDecoder(cuContext, true, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));
    nWidth=0;
    nHeight=0;
    nFrameSize=0;
    anSize=new int[7]{ 0, 3, 3, 4, 4, 8, 8 };
    nVideoBytes = 0;
    nFrameReturned = 0;
    nFrame = 0;
    iMatrix = 0;
    pVideo=nullptr;

    // nPitch=1280*4;
    // cv_img_bgr.create(720,1280,CV_8UC3);

    // cudaMalloc((void**)&tmp_bgr,sizeof(uint8_t)*size_frame_bgr);
    // gInstance=new FramePresenterGLX(720,1280);

    // bool initFlag=true;
    // if(dec->GetHeight()>0 && dec->GetWidth()>0 && initFlag){
    //     initFrame(dec->GetHeight(),dec->GetWidth());
    //     initFlag=false;
    // }

    initFrame(720,1280);

}


void DecodeGL::initFrame(int rows,int cols){
    height=rows;
    width=cols;
    channel=3;
    size_frame_bgra=height*width*4;
    size_frame_bgr=height*width*3;
    cudaMalloc((void**)&tmp,sizeof(uint8_t)*rows*cols*4);

    // cv_img_bgr.create(720,1280,CV_8UC3);

    // cudaMalloc((void**)&tmp_bgr,sizeof(uint8_t)*size_frame_bgr);

    
}


/**
*   @brief  Function to copy image data from CUDA device pointer to host buffer
*   @param  dpSrc   - CUDA device pointer which holds decoded raw frame
*   @param  pDst    - Pointer to host buffer which acts as the destination for the copy
*   @param  nWidth  - Width of decoded frame
*   @param  nHeight - Height of decoded frame
*/
void DecodeGL::GetImage(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight)
{
    CUDA_MEMCPY2D m = { 0 };
    m.WidthInBytes = nWidth;
    m.Height = nHeight;
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)dpSrc;
    m.srcPitch = m.WidthInBytes;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
    m.dstPitch = m.WidthInBytes;
    cuMemcpy2D(&m);
}

// enum OutputFormat
// {
//     native = 0, bgrp, rgbp, bgra, rgba, bgra64, rgba64
// };



std::string DecodeGL::GetSupportedFormats()
{
    std::vector<std::string> vstrOutputFormatName =
    {
        "native", "bgrp", "rgbp", "bgra", "rgba", "bgra64", "rgba64"
    };

    std::ostringstream oss;
    for (auto& v : vstrOutputFormatName)
    {
        oss << " " << v;
    }

    return oss.str();
}

int DecodeGL::decodeFrame(
    uint8_t *bgr_frame,
    uint8_t *bgra_frame,
    CUdeviceptr &pTmpImage,
    cudaStream_t &stream,
    int idx,
    int idj
    ){
    
    // OutputFormat eOutputFormat = bgra;
    

    // nvtxRangePush(__FUNCTION__);
    std::string str1="decodeFrame-"+std::to_string(idx)+'-'+std::to_string(idj);
    // const char* c1=str1.c_str();
    // nvtxRangePushA(str1.c_str());

    

    demuxer->Demux(&pVideo, &nVideoBytes);

    

    // CUcontext cuContext = NULL;
    // createCudaContext(&cuContext, 1, CU_CTX_SCHED_BLOCKING_SYNC);
    // NvDecoder *dec=new NvDecoder(cuContext, true, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));

    // std::cout<<"Video info1 "<<std::endl;
    nFrameReturned = dec->Decode(pVideo, nVideoBytes);
    if(!nVideoBytes && start_flag){
        return -2;
    }
    // std::cout<<"nFrame "<<nFrame<<std::endl;
    // std::cout<<"nFrameReturned "<<nFrameReturned<<std::endl;

    // CUdeviceptr pTmpImage;
    if (!nFrame && nFrameReturned)
    {
        // std::cout<<"Video info "<<std::endl;
        LOG(INFO) << dec->GetVideoInfo();
        // Get output frame size from decoder
        nWidth = dec->GetWidth(); nHeight = dec->GetHeight();
        nFrameSize = eOutputFormat == native ? dec->GetFrameSize() : nWidth * nHeight * anSize[eOutputFormat];
        std::unique_ptr<uint8_t[]> pTemp(new uint8_t[nFrameSize]);
        pImage = std::move(pTemp);
        std::cout<<"nWidth-> "<<nWidth<<std::endl;
        std::cout<<"nHeight-> "<<nHeight<<std::endl;
        std::cout<<"anSize[eOutputFormat]-> "<<anSize[eOutputFormat]<<std::endl;
        cuMemAlloc(&pTmpImage, nWidth * nHeight * anSize[eOutputFormat]);
        nFrame=1;
        start_flag=true;
    }

    // std::cout<<"nFrameReturned-->"<<nFrameReturned<<" : "<<url<<std::endl;

    if(nFrameReturned<1){
        return -1;
    }
    if(nFrameReturned>1){
        nFrameReturned=1;
    }
    nvtxRangePushA(str1.c_str());

    // nvtxRangePushA("GetFrame");
    iMatrix = dec->GetVideoFormatInfo().video_signal_description.matrix_coefficients;
    pFrame = dec->GetFrame();
    // bgr_frame = dec->GetFrame();

    // nvtxRangePop();

    // gInstance->GetDeviceFrameBuffer(&pTmpImage, &nPitch);
    

    if (dec->GetBitDepth() == 8) {
        // std::string str2="Frame->BGR --point-"+std::to_string(idx);
        // nvtxMark(str2);
        std::string str3="Format frame-"+std::to_string(idx)+'-'+std::to_string(idj);
        nvtxRangePushA(str3.c_str());
        if (dec->GetOutputFormat() == cudaVideoSurfaceFormat_YUV444){
            YUV444ToColor32<BGRA32>(pFrame, dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);
            // YUV444ToColor32<BGRA32>(bgr_frame, dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);
            std::cout<<"YUV444ToColor32"<<std::endl;
        
        }
        else{
            Nv12ToColor32Async<BGRA32>(
                pFrame, dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix,stream);
            // Nv12ToColor32<BGRA32>(bgr_frame, dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);
            // std::cout<<"Nv12ToColor32Async"<<std::endl;
        }

        //BGRA processing
        // std::cout<<dec->GetWidth()<<" : "<<dec->GetHeight()<<std::endl;
        resize_nearest_gpu((uint8_t*)pTmpImage,tmp,dec->GetWidth(),dec->GetHeight(),1280,720,stream);
        bgra2bgrAsync(tmp,bgr_frame,720,1280,stream);
        bgr2bgraAsync(bgr_frame,bgra_frame,720,1280,stream);
        

        // nvtxRangePop();
    }
    else
    {
        switch (eOutputFormat) {
        case bgra:
            if (dec->GetOutputFormat() == cudaVideoSurfaceFormat_YUV444_16Bit)
                YUV444P16ToColor32<BGRA32>(pFrame, 2 * dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);
            else
                P016ToColor32<BGRA32>(pFrame, 2 * dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);

            break;
        case rgba:
            if (dec->GetOutputFormat() == cudaVideoSurfaceFormat_YUV444_16Bit)
                YUV444P16ToColor32<RGBA32>(pFrame, 2 * dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);
            else
                P016ToColor32<RGBA32>(pFrame, 2 * dec->GetWidth(), (uint8_t*)pTmpImage, 4 * dec->GetWidth(), dec->GetWidth(), dec->GetHeight(), iMatrix);
            break;
        }

        //BGRA processing
        // std::cout<<dec->GetWidth()<<" : "<<dec->GetHeight()<<std::endl;
        resize_nearest_gpu((uint8_t*)pTmpImage,tmp,dec->GetWidth(),dec->GetHeight(),1280,720,stream);
        bgra2bgrAsync(tmp,bgr_frame,720,1280,stream);
        bgr2bgraAsync(bgr_frame,bgra_frame,720,1280,stream);
    }

    nvtxRangePop();

    // gInstance->ReleaseDeviceFrameBuffer();
    
    // nFrame += nFrameReturned;

    // if (pTmpImage) {
    //     cuMemFree(pTmpImage);
    // }

    return 1;
    
}

