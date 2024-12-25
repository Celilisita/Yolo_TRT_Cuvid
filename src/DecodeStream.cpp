#include "Decode.h"
// #include "CenterNetTRT.h"

void showPrint(CUcontext context){
    std::cout<<"Context-> "<<context<<std::endl;
    std::cout<<"&Context-> "<<&context<<std::endl;
}

void runStream(
    CUcontext cuContext,
    FramePresenterGLX *glx,
    DecodeGL *dec,
    CenterNetTRT *infer,
    std::string url,
    uint8_t *bgr_frame,
    uint8_t *bgra_frame,
    CUdeviceptr &pTmpImage,
    cudaStream_t &stream,
    int nPitch,
    int idx
){
    cuCtxSetCurrent(cuContext);
    // CUcontext cuContext;
    // createCudaContext(&cuContext,0,0);

    // std::cout<<"cuContext--> "<<cuContext<<std::endl;
    // std::cout<<"&cuContext--> "<<&cuContext<<std::endl;
    // // showPrint(cuContext);

    // DecodeGL *dec=new DecodeGL();
    dec->set_url(url,cuContext);
    dec->dec->m_cuvidStream=stream;
    // FramePresenterGLX *glx=new FramePresenterGLX(1280,720);
    // glx=new FramePresenterGLX(1280,720);

    // CUdeviceptr ceptr;
    // ceptr=0;

    // CenterNetTRT *infer=new CenterNetTRT(1);

    // std::cout<<"sub1 thread Context-> "<<context<<std::endl;
    // std::cout<<"sub2 thread Context-> "<<&context<<std::endl;
    int size_bgra=1280*720*4;
    for(int i=0;i<300;i++){
        std::string str1="Camera-"+std::to_string(idx);
        nvtxRangePushA(str1.c_str());
        

        dec->decodeFrame(bgr_frame,bgra_frame,pTmpImage,stream,idx,i);
        infer->detect_direct_idx(idx,bgr_frame,stream,1.875f,2.0f);
        // infer->detect_direct_idx(0,bgr_frame,stream,1.875f,2.0f);

        // glx->GetDeviceFrameBuffer(&pTmpImage, &nPitch);
        // cudaMemcpyAsync((uint8_t*)pTmpImage,bgra_frame,sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToDevice,stream);
        // glx->ReleaseDeviceFrameBuffer();
        

        nvtxRangePop();
    }
}



int Process1(){

    //1.decode

    //2.process

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    ck(cuInit(0));
    int iGpu;
    int nGpu=0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu)
    {
        std::ostringstream err;
        err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
        throw std::invalid_argument(err.str());
    }

    CUcontext cuContext;
    cuContext=NULL;
    createCudaContext(&cuContext,iGpu,0);


    std::string url1="rtsp://admin:ld123456@192.168.200.112:554";
    // std::string url1="/home/dl/Downloads/LG303_2.mp4";
    std::string url2="rtsp://admin:ld123456@192.168.200.111:554";
    std::string url3="rtsp://admin:ld123456@192.168.200.110:554";
    std::string url4="rtsp://admin:ld123456@192.168.200.109:554";
    std::string url5="rtsp://admin:ld123456@192.168.200.108:554";
    std::string url6="rtsp://admin:ld123456@192.168.200.107:554";
    std::string url7="rtsp://admin:ld123456@192.168.200.106:554";
    std::string url8="rtsp://admin:ld123456@192.168.200.105:554";
    std::string url9="rtsp://admin:ld123456@192.168.200.104:554";
    std::string url10="rtsp://admin:ld123456@192.168.200.103:554";
    std::string url11="rtsp://admin:ld123456@192.168.200.102:554";
    std::string url12="rtsp://admin:ld123456@192.168.200.101:554";
    std::string url13="rtsp://admin:ld123456@192.168.200.113:554";
    std::string url14="rtsp://admin:ld123456@192.168.200.118:554";
    std::string url15="rtsp://admin:ld123456@192.168.200.114:554";
    std::string url16="rtsp://admin:ld123456@192.168.200.119:554";
    std::string url17="rtsp://admin:ld123456@192.168.200.121:554";
    std::string url18="rtsp://admin:ld123456@192.168.200.122:554";
    std::string url19="rtsp://admin:ld123456@192.168.200.123:554";
    std::string url20="rtsp://admin:ld123456@192.168.200.124:554";

    std::vector<std::string> url_vec;
    url_vec.push_back(url1);
    url_vec.push_back(url2);
    url_vec.push_back(url3);
    url_vec.push_back(url4);
    url_vec.push_back(url5);
    url_vec.push_back(url6);
    url_vec.push_back(url7);
    url_vec.push_back(url8);
    url_vec.push_back(url9);
    url_vec.push_back(url10);
    url_vec.push_back(url11);
    url_vec.push_back(url12);
    url_vec.push_back(url13);
    url_vec.push_back(url14);
    url_vec.push_back(url15);
    url_vec.push_back(url16);
    url_vec.push_back(url17);
    url_vec.push_back(url18);
    url_vec.push_back(url19);
    url_vec.push_back(url20);

    int n_stream=url_vec.size();
    int size_bgr=1280*720*3;
    int size_bgra=1280*720*4;

    cv::Mat cv_img_bgra;
    cv_img_bgra.create(720,1280,CV_8UC4);

    DecodeGL **dec_vec;
    FramePresenterGLX **glx_vec;

    cudaStream_t *streams;
    uint8_t **bgr_frames;
    uint8_t **bgra_frames;

    CUdeviceptr *ptrs;
    CUdeviceptr *qtrs;
    int* nPitch;

    streams=(cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
    bgr_frames=(uint8_t**)malloc(n_stream*sizeof(uint8_t*));
    bgra_frames=(uint8_t**)malloc(n_stream*sizeof(uint8_t*));

    ptrs=(CUdeviceptr*)malloc(n_stream*sizeof(CUdeviceptr));
    // qtrs=(CUdeviceptr*)malloc(n_stream*sizeof(CUdeviceptr));
    nPitch=(int*)malloc(n_stream*sizeof(int));

    dec_vec=(DecodeGL**)malloc(sizeof(DecodeGL*)*n_stream);
    glx_vec=(FramePresenterGLX**)malloc(sizeof(FramePresenterGLX*)*n_stream);

    for(int i=0;i<n_stream;i++){
        DecodeGL *dec=new DecodeGL();
        dec_vec[i]=dec;
        // dec_vec[i]->init(url_vec[i]);
        dec_vec[i]->set_url(url_vec[i],cuContext);

        cudaStreamCreate(&streams[i]);
        cudaMalloc((void**)&bgr_frames[i],sizeof(uint8_t)*size_bgr);
        cudaMalloc((void**)&bgra_frames[i],sizeof(uint8_t)*size_bgra);

        dec_vec[i]->dec->m_cuvidStream=streams[i];

        // FramePresenterGLX *glx=new FramePresenterGLX(720,1280);
        FramePresenterGLX *glx=new FramePresenterGLX(1280,720);
        glx_vec[i]=glx;
        ptrs[i]=i;
        // qtrs[i]=i;
        nPitch[i]=1280*4;

        // glx_vec[i]->GetDeviceFrameBuffer(&ptrs[i], &nPitch[i]);

    }

    CUdeviceptr ceptr;
    // ceptr=0;
    
    // DecodeGL *decode_g=new DecodeGL();
    // decode_g->init(url1);

    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // uint8_t *bgr_frame;
    // cudaMalloc((void**)&bgr_frame,sizeof(uint8_t)*size_bgr);`

    for(int i=0;i<400;i++){
        std::string str1="Iter-"+std::to_string(i);
        nvtxRangePushA(str1.c_str());
        for(int j=0;j<n_stream;j++){
            std::string str1="Camera-"+std::to_string(j);
            nvtxRangePushA(str1.c_str());
            // glx_vec[j]->GetDeviceFrameBuffer(&ptrs[j], &nPitch[j]);
            // glx_vec[j]->GetDeviceFrameBuffer(&ceptr, &nPitch[j]);
            std::string str2="Drawing Prepare-"+std::to_string(j);
            nvtxRangePushA(str2.c_str());
            glx_vec[j]->GetDeviceFrameBuffer(&ptrs[j], &nPitch[j]);

            nvtxRangePop();

            dec_vec[j]->decodeFrame(bgr_frames[j],bgra_frames[j],ptrs[j],streams[j],i,j);
            // glx_vec[j]->GetDeviceFrameBuffer(&ptrs[j], &nPitch[j]);

            nvtxRangePushA("OpenGL Drawing");

            cudaMemcpyAsync((uint8_t*)ptrs[j],bgra_frames[j],sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToDevice,streams[j]);
            // cudaMemcpy((uint8_t*)ceptr,bgra_frames[j],sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToDevice);
            // cudaMemcpy(cv_img_bgra.data,bgra_frames[j],sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToHost);
            // cv::imshow("Debug-"+std::to_string(j),cv_img_bgra);
            // cv::waitKey(10);
            glx_vec[j]->ReleaseDeviceFrameBuffer();
            nvtxRangePop();

            nvtxRangePop();

        }

        // for(int j=0;j<2;j++){
            
        //     std::string str2="Drawing Prepare-"+std::to_string(j);
        //     nvtxRangePushA(str2.c_str());
        //     glx_vec[j]->GetDeviceFrameBuffer(&ptrs[j], &nPitch[j]);

        //     // nvtxRangePop();

        //     // dec_vec[j]->decodeFrame(bgr_frames[j],bgra_frames[j],ptrs[j],streams[j],i,j);
        //     // glx_vec[j]->GetDeviceFrameBuffer(&ptrs[j], &nPitch[j]);

        //     // nvtxRangePushA("OpenGL Drawing");

        //     cudaMemcpyAsync((uint8_t*)ptrs[j],bgra_frames[j],sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToDevice,streams[j]);

        //     // nvtxRangePop();
        //     glx_vec[j]->ReleaseDeviceFrameBuffer();
        //     nvtxRangePop();

        // }

        nvtxRangePop();

        // cudaEventSynchronize(stop);
        
    }

    // for(int j=0;j<n_stream;j++){
    //     FramePresenterGLX *glx=glx_vec[j];
    //     DecodeGL *dec=dec_vec[j];
    //     CUdeviceptr pTmpImage=ptrs[j];
    //     uint8_t *bgr_frame=bgr_frames[j];
    //     uint8_t *bgra_frame=bgra_frames[j];
    //     cudaStream_t stream=streams[j];
    //     int Pitch=nPitch[j];

    //     std::thread([glx,dec,&pTmpImage,
    //         bgr_frame,bgra_frame,
    //         &stream,&Pitch,j]{
    //             int size_bgra=1280*720*4;
    //             for(int i=0;i<40000;i++){
    //                 dec->decodeFrame(bgr_frame,bgra_frame,pTmpImage,stream,j);
    //                 glx->GetDeviceFrameBuffer(&pTmpImage, &Pitch);
    //                 cudaMemcpyAsync((uint8_t*)pTmpImage,bgra_frame,sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToDevice,stream);
    //                 glx->ReleaseDeviceFrameBuffer();
    //             }
    //         }).detach();

    //     // std::thread(
    //     //     runStream,glx_vec[j],
    //     //     dec_vec[j],bgr_frames[j],
    //     //     bgra_frames[j],ptrs[j],
    //     //     streams[j],nPitch[j],j).detach();
    // }
        
    
    for(int i=0;i<n_stream;i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(streams);
    
    // delete decode_g;


    return 0;
}

int Process2(){



    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    ck(cuInit(0));
    int iGpu;
    int nGpu=0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu)
    {
        std::ostringstream err;
        err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
        throw std::invalid_argument(err.str());
    }

    


    std::string url1="rtsp://admin:ld123456@192.168.200.112:554";
    // std::string url1="/home/dl/Downloads/LG303_2.mp4";
    std::string url2="rtsp://admin:ld123456@192.168.200.111:554";
    std::string url3="rtsp://admin:ld123456@192.168.200.110:554";
    std::string url4="rtsp://admin:ld123456@192.168.200.109:554";
    std::string url5="rtsp://admin:ld123456@192.168.200.108:554";
    std::string url6="rtsp://admin:ld123456@192.168.200.107:554";
    std::string url7="rtsp://admin:ld123456@192.168.200.106:554";
    std::string url8="rtsp://admin:ld123456@192.168.200.105:554";
    std::string url9="rtsp://admin:ld123456@192.168.200.104:554";
    std::string url10="rtsp://admin:ld123456@192.168.200.103:554";
    std::string url11="rtsp://admin:ld123456@192.168.200.102:554";
    std::string url12="rtsp://admin:ld123456@192.168.200.101:554";
    std::string url13="rtsp://admin:ld123456@192.168.200.113:554";
    std::string url14="rtsp://admin:ld123456@192.168.200.118:554";
    std::string url15="rtsp://admin:ld123456@192.168.200.114:554";
    std::string url16="rtsp://admin:ld123456@192.168.200.119:554";
    std::string url17="rtsp://admin:ld123456@192.168.200.121:554";
    std::string url18="rtsp://admin:ld123456@192.168.200.122:554";
    std::string url19="rtsp://admin:ld123456@192.168.200.123:554";
    std::string url20="rtsp://admin:ld123456@192.168.200.124:554";

    std::vector<std::string> url_vec;
    url_vec.push_back(url1);
    url_vec.push_back(url2);
    // url_vec.push_back(url3);
    // url_vec.push_back(url4);
    // url_vec.push_back(url5);
    // url_vec.push_back(url6);
    // url_vec.push_back(url7);
    // url_vec.push_back(url8);
    // url_vec.push_back(url9);
    // url_vec.push_back(url10);
    // url_vec.push_back(url11);
    // url_vec.push_back(url12);
    // url_vec.push_back(url13);
    // url_vec.push_back(url14);
    // url_vec.push_back(url15);
    // url_vec.push_back(url16);
    // url_vec.push_back(url17);
    // url_vec.push_back(url18);
    // url_vec.push_back(url19);
    // url_vec.push_back(url20);
    // url_vec.push_back(url1);
    // url_vec.push_back(url2);
    // url_vec.push_back(url3);
    // url_vec.push_back(url4);
    // url_vec.push_back(url5);
    // url_vec.push_back(url6);
    // url_vec.push_back(url7);
    // url_vec.push_back(url8);
    // url_vec.push_back(url9);
    // url_vec.push_back(url10);
    // url_vec.push_back(url11);
    // url_vec.push_back(url12);
    // url_vec.push_back(url13);
    // url_vec.push_back(url14);
    // url_vec.push_back(url15);
    // url_vec.push_back(url16);
    // url_vec.push_back(url17);
    // url_vec.push_back(url18);
    // url_vec.push_back(url19);
    // url_vec.push_back(url20);
    // url_vec.push_back(url1);
    // url_vec.push_back(url2);
    // url_vec.push_back(url3);
    // url_vec.push_back(url4);
    // url_vec.push_back(url5);
    // url_vec.push_back(url6);
    // url_vec.push_back(url7);
    // url_vec.push_back(url8);
    // url_vec.push_back(url9);
    // url_vec.push_back(url10);
    // url_vec.push_back(url11);
    // url_vec.push_back(url12);
    // url_vec.push_back(url13);
    // url_vec.push_back(url14);
    // url_vec.push_back(url15);
    // url_vec.push_back(url16);
    // url_vec.push_back(url17);
    // url_vec.push_back(url18);
    // url_vec.push_back(url19);
    // url_vec.push_back(url20);

    int n_stream=url_vec.size();
    int size_bgr=1280*720*3;
    int size_bgra=1280*720*4;

    CUcontext cuContext;
    cuContext=NULL;
    createCudaContext(&cuContext,iGpu,0);

    // std::cout<<"main cuContext--> "<<cuContext<<std::endl;
    // std::cout<<"main &cuContext--> "<<&cuContext<<std::endl;


    // CUcontext* cuContexts;
    // cuContexts=(CUcontext*)malloc(sizeof(CUcontext)*n_stream);
    

    // std::cout<<"main thread Context-> "<<cuCtxGetCurrent(&cuContext)<<std::endl;

    // cv::Mat cv_img_bgra;
    // cv_img_bgra.create(720,1280,CV_8UC4);

    DecodeGL **dec_vec;
    FramePresenterGLX **glx_vec;

    cudaStream_t *streams;
    uint8_t **bgr_frames;
    uint8_t **bgra_frames;

    CUdeviceptr *ptrs;
    CUdeviceptr *qtrs;
    int* nPitch;

    streams=(cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
    bgr_frames=(uint8_t**)malloc(n_stream*sizeof(uint8_t*));
    bgra_frames=(uint8_t**)malloc(n_stream*sizeof(uint8_t*));

    ptrs=(CUdeviceptr*)malloc(n_stream*sizeof(CUdeviceptr));
    // qtrs=(CUdeviceptr*)malloc(n_stream*sizeof(CUdeviceptr));
    nPitch=(int*)malloc(n_stream*sizeof(int));

    dec_vec=(DecodeGL**)malloc(sizeof(DecodeGL*)*n_stream);
    glx_vec=(FramePresenterGLX**)malloc(sizeof(FramePresenterGLX*)*n_stream);

    //CenterNet
    CenterNetTRT *infer=new CenterNetTRT(url_vec.size());
    // infer->img_num=url_vec.size();

    for(int i=0;i<n_stream;i++){

        // createCudaContext(&cuContexts[i],iGpu,0);
        // std::cout<<"main thread Context-> "<<cuContexts[i]<<std::endl;

        DecodeGL *dec=new DecodeGL();
        dec_vec[i]=dec;
        // // dec_vec[i]->init(url_vec[i]);
        // dec_vec[i]->set_url(url_vec[i],cuContexts[i]);

        cudaStreamCreate(&streams[i]);
        cudaMalloc((void**)&bgr_frames[i],sizeof(uint8_t)*size_bgr);
        cudaMalloc((void**)&bgra_frames[i],sizeof(uint8_t)*size_bgra);

        // dec_vec[i]->dec->m_cuvidStream=streams[i];

        // FramePresenterGLX *glx=new FramePresenterGLX(720,1280);
        // FramePresenterGLX *glx=new FramePresenterGLX(1280,720);
        // glx_vec[i]=glx;
        ptrs[i]=i;
        // qtrs[i]=i;
        nPitch[i]=1280*4;

        // glx_vec[i]->GetDeviceFrameBuffer(&ptrs[i], &nPitch[i]);

    }

    // void runStream(
    //     FramePresenterGLX *glx,
    //     DecodeGL *dec,
    //     uint8_t *bgr_frame,
    //     uint8_t *bgra_frame,
    //     CUdeviceptr &pTmpImage,
    //     cudaStream_t &stream,
    //     int nPitch,
    //     int idx
    // )

    for(int j=0;j<n_stream;j++){
        // FramePresenterGLX *glx=glx_vec[j];
        // DecodeGL *dec=dec_vec[j];
        // CUdeviceptr pTmpImage=ptrs[j];
        // uint8_t *bgr_frame=bgr_frames[j];
        // uint8_t *bgra_frame=bgra_frames[j];
        // // cudaStream_t stream=streams[j];
        // int Pitch=nPitch[j];

        std::thread t(&runStream,
            cuContext,
            // cuContexts[j],
            glx_vec[j],
            dec_vec[j],
            infer,
            url_vec[j],
            bgr_frames[j],
            bgra_frames[j],
            std::ref(ptrs[j]),
            std::ref(streams[j]),
            nPitch[j],
            j
        );
        t.detach();
    }
    
    int iter=0;
    while(iter<30){
        sleep(1);
        iter++;
        std::cout<<"iter-> "<<iter<<std::endl;
    }
    
    for(int i=0;i<n_stream;i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(streams);
    
    // delete decode_g;


    return 0;
}

int main(){
    Process2();
    return 0;
}