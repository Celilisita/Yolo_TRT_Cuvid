#include "yolov5_trt.h"
#include "Decode.h"
#include "bbox_calculate.h"

#include "Affine.h"

std::mutex lock_;
std::mutex lock_end;
const int limit_ =3;
std::map<int,std::queue<Job>> thread_points;
// std::map<int,std::queue<int>> thread_rects;
std::map<int,int> thread_display_idx;
std::map<int,int> thread_display_rect;
std::condition_variable condition;

bool end_infer=false;

float *rect_ptr;
const int max_rect=1000;

bool isPush=false;
bool isShow=false;

void decode_frame(
    CUcontext &context,
    cudaStream_t stream,
    DecodeGL *dec,
    FramePresenterD3D11 *presenter,
    std::string pushUrl,
    uint8_t *bgra_frames,
    uint8_t *bgr_frames,
    uint8_t *render_surfaces,
    uint8_t *yuvs,
    bool *end_arr,
    int thread_id
){
    // double total_s, total_e, total_res;
    // double t_s, t_e, t_res;
    // total_s = cv::getTickCount();

    // int64_t delay=1000/1000;
    int64_t delay=1;

    // std::string url="06_720p.mp4";
    //FFmpeg streamPush
    // std::string pushUrl = "rtmp://127.0.0.1:1935/live1";


    cuCtxPushCurrent(context);



    int nPitch=1280*4;
    int nWidth=1280;
    int nHeight=720;

    CUdeviceptr ptr=0;

    cuMemAlloc(&ptr,nWidth*nHeight*4);

    int size_bgra=nWidth*nHeight*4;
    int size_bgr=nWidth*nHeight*3;
    int size_yuv=nWidth*nHeight*3/2;
    int idx;

    // bool isPush=false;
    if(pushUrl.length()>1){
        isPush=true;
    }

    //Another streampush
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    NvEncoderInitParam pEncodeCLIOptions;

    NvEncoderCuda enc(context, nWidth, nHeight, eFormat, 3, false, false, false);
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    enc.CreateDefaultEncoderParams(
        &initializeParams, 
        pEncodeCLIOptions.GetEncodeGUID(), 
        pEncodeCLIOptions.GetPresetGUID(), 
        pEncodeCLIOptions.GetTuningInfo()
    );

    pEncodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    enc.CreateEncoder(&initializeParams);
    int nHostFrameSize = enc.GetFrameSize();

    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nHostFrameSize]);
    CUdeviceptr dpBgraFrame = 0;
    ck(cuMemAlloc(&dpBgraFrame, nWidth * nHeight * 8));

    FFmpegStreamer *streamer;
    if(isPush){
        streamer=new FFmpegStreamer(
            pEncodeCLIOptions.IsCodecH264() ? AV_CODEC_ID_H264 : pEncodeCLIOptions.IsCodecHEVC() ? AV_CODEC_ID_HEVC : AV_CODEC_ID_AV1, 
            nWidth, nHeight, int(1000/delay), pushUrl.data());
    }
    std::vector<std::vector<uint8_t>> vPacket;

    int nVideoBytes=0;

    int i=0;
    double sum=0;
    while(true){
        // t_s = cv::getTickCount();
        auto start_cpu = std::chrono::steady_clock::now();
        Job job;

        {
            std::unique_lock<std::mutex> lock_product(lock_);
            idx=i%limit_+thread_id*limit_;
            // dec->decodeFrame(bgr_frames+size_bgr*idx,bgra_frames+size_bgra*idx,ptr,stream,0,i);
            nVideoBytes=dec->decodeFrame(bgr_frames+size_bgr*idx,bgra_frames+size_bgra*idx,ptr,stream,0,i++);

            // printf("%d decode frames in %d\n",idx,i);
            condition.wait(lock_product,[&idx](){
                return thread_points[idx].size()<limit_;
            });

            job.pro.reset(new std::promise<int>());
            // // job.input="Thread["+std::to_string(thread_id)+"]";
            job.device_idx=idx;

            // que_jobs.push(job);
            thread_points[thread_id].push(job); 

        }
        
        auto fu=job.pro->get_future();
        auto result=fu.get();
        int infer_rect_num=result;


        // trt->inference(bgr_frames+size_bgr*thread_id,stream);

        if(isShow){

        
            presenter->PresentRectFrame(rect_ptr+idx*max_rect*6, infer_rect_num);
            // // presenter->PresentRectFrame(trt->pRect, trt->p_num);

            presenter->PresentDeviceFrame(bgra_frames+size_bgra*idx,render_surfaces+size_bgra*idx,nPitch,delay);

            Bgra2YuvAsync(render_surfaces+size_bgra*idx,yuvs+size_yuv*idx,nHeight,nWidth,stream);

            // cudaStreamSynchronize(stream);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        
        if(nVideoBytes==-2){
            break;
        }

        if(isPush){
        
            const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();
            
            NvEncoderCuda::CopyToDeviceFrame(
                context, yuvs+size_yuv*idx, 0, 
                (CUdeviceptr)encoderInputFrame->inputPtr,
                (int)encoderInputFrame->pitch,
                enc.GetEncodeWidth(),
                enc.GetEncodeHeight(),
                CU_MEMORYTYPE_DEVICE,
                encoderInputFrame->bufferFormat,
                encoderInputFrame->chromaOffsets,
                encoderInputFrame->numChromaPlanes);
            
            enc.EncodeFrame(vPacket);

            for (std::vector<uint8_t> &packet : vPacket) {
                streamer->Stream(packet.data(), (int)packet.size(), i);
            }
        }
        auto end_cpu = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed_cpu = end_cpu - start_cpu; //
        // std::cout<< "Resize CPU Cost: "  << elapsed_cpu.count()/1000.0f << " ms" << std::endl; 
        // t_e = cv::getTickCount();
        // t_res = (t_e - t_s) / cv::getTickFrequency();
        sum+=elapsed_cpu.count()/1000.0f;
        std::cout <<"\r"<<"thread_id->[ "<<thread_id<< " ] iter "<<i+1<<" C:[ "<<idx<<" ]"<<" Get:["<<infer_rect_num<<"] rects! "
            <<" cost--> " << sum/(i+1)  << " ms" << std::flush;
        // std::cout <<"thread_id->[ "<<thread_id<< " ] iter "<<i+1<<" C:[ "<<idx<<" ]"<<" Get:["<<infer_rect_num<<"] rects! "
        //     <<" cost--> " << sum/(i+1)  << " ms" << std::endl;
        // std::cout <<"thread_id->[ "<<thread_id<< " ] iter "<<i+1<<" C:[ "<<idx<<" ]"
        //     <<" cost--> " << t_res * 1000 << " ms" << std::endl;

        // std::this_thread::yield();
    }

    cuMemFree(ptr);
    enc.EndEncode(vPacket);
    enc.DestroyEncoder();

    // delete presenter;

    cuCtxPopCurrent(NULL);

    {
        std::lock_guard<std::mutex> lock_thread(lock_);
        end_arr[thread_id]=true;
        printf("***Thread: %d Finish!***\n",thread_id);
    }
    
}

void infer_trt(
    CUcontext &context,
    cudaStream_t *streams,
    Yolov5TRT *trt,
    uint8_t *bgr_frames
){

    cuCtxPushCurrent(context);

    int nPitch=1280*4;
    int nWidth=1280;
    int nHeight=720;
    int size_bgr=nWidth*nHeight*3;

    while(!end_infer){
        std::map<int,std::queue<Job>>::iterator iter;
        for(iter=thread_points.begin();iter!=thread_points.end();iter++){
            int id=iter->first;
            if(!thread_points[id].empty()){
                {
                    std::lock_guard<std::mutex> lock_infer(lock_);
                    // if(end_infer){
                    //     break;
                    // }
                    auto job=thread_points[id].front();
                    
                    thread_points[id].pop();
                    condition.notify_one();

                    int idx=job.device_idx;
                    trt->inference(bgr_frames+size_bgr*idx,streams[id]);

                    cudaMemcpyAsync(rect_ptr+max_rect*idx*6,trt->pRect,sizeof(float)*6*trt->p_num,cudaMemcpyHostToHost,streams[id]);
                    // auto result=job.input+"--- infer thread:"+std::to_string(id);

                    // thread_rects[id].push
                    thread_display_idx[id]=idx;
                    thread_display_rect[id]=trt->p_num;

                    job.pro->set_value(trt->p_num);
                    // job.pnum=trt->p_num;
                    // printf("infer img in Thread:[%d]->[%d] have:[%d] rectangles remain: %d pictures\n",id,idx,trt->p_num,thread_points[id].size());
                    
                    
                }
                // std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // else{
            //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // }
            std::this_thread::yield();
        }
        // {
        //     std::lock_guard<std::mutex> end_lock(lock_end);
        //     if(end_infer){
        //         break;
        //     }
        // }
        
        // std::this_thread::yield();
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    printf("+++++++++++++++++++++++++++++++\n");
    printf("Close infer worker!\n");
    printf("+++++++++++++++++++++++++++++++\n");

    cuCtxPopCurrent(NULL);
}


void aysnc_process(std::vector<std::string> &url_vec,std::vector<std::string> &push_vec){

    cuInit(0);
    int nGpu=0;
    int device_id=1;
    cuDeviceGetCount(&nGpu);
    if(nGpu<1){
        std::cerr<<"No avaviable GPU!"<<std::endl;

    }
    if (nGpu == 1) {
        device_id = 0;
    }

    int n_stream=url_vec.size();
    
    // cudaSetDevice(device_id);

    //set device:0 to display
    //Display context
    CUdevice cuDevicev0 = 0;
    cuDeviceGet(&cuDevicev0, 0);
    CUcontext context0=NULL;
    cuCtxCreate(&context0, 0, cuDevicev0);
    // cudaStream_t d3d_stream;
    // cudaStreamCreate(&d3d_stream);

    // CUcontext *contexts=(CUcontext*)malloc(sizeof(CUcontext)*n_stream);
    cudaStream_t *d3d_streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*n_stream);
    for(int i=0;i<n_stream;i++){
        // CUcontext context_tmp=NULL;
        // cuCtxCreate(&context_tmp, 0, cuDevicev0);
        cudaStream_t d3d_stream;
        cudaStreamCreate(&d3d_stream);
        // contexts[i]=context_tmp;
        d3d_streams[i]=d3d_stream;
    }

    CUcontext context=NULL;
    // createCudaContext(&context,device_id,0);
    CUdevice cuDevice = 0;
    cuDeviceGet(&cuDevice, device_id);
    char szDeviceName[80];
    cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    cuCtxCreate(&context, 0, cuDevice);

    

    // int n_stream=url_vec.size();


    cudaStream_t* streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*n_stream);

    for(int i=0;i<n_stream;i++){

        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    }


    
    int nPitch=1280*4;
    int nWidth=1280;
    int nHeight=720;
    // int max_rect=1000;


    DecodeGL **dec_list=(DecodeGL**)malloc(sizeof(DecodeGL*)*n_stream);
    for(int i=0;i<n_stream;i++){
        DecodeGL *dec=new DecodeGL();
        dec_list[i]=dec;
        dec_list[i]->set_url(url_vec[i],context);
        dec_list[i]->dec->m_cuvidStream=streams[i];
    }

    Yolov5TRT *trt=new Yolov5TRT(device_id);
    trt->init(720,1280);


    FramePresenterD3D11 **presenters=(FramePresenterD3D11**)malloc(sizeof(FramePresenterD3D11*)*n_stream);
    for(int i=0;i<n_stream;i++){
        // presenters[i]=new FramePresenterD3D11(context,nWidth,nHeight,streams[i]);
        // FramePresenterD3D11 *presenter=new FramePresenterD3D11(context,nWidth,nHeight,streams[i]);
        FramePresenterD3D11 *presenter=new FramePresenterD3D11(
            context0,nWidth,nHeight,const_cast<char*>(url_vec[i].c_str()),d3d_streams[i]);
        presenter->setWinName(url_vec[i]);
        // // presenter->colorInit(engines[i]->pColors, engines[i]->class_num);
        presenter->colorInit(trt->pColors, trt->class_num);
        presenters[i]=presenter;
    }

    bool *end_arr=(bool*)malloc(sizeof(bool)*n_stream);
    memset(end_arr,false,sizeof(bool)*n_stream);


    uint8_t *bgra_frames;
    uint8_t *bgr_frames;
    uint8_t *render_surfaces;
    uint8_t *yuvs;
    int size_bgra=nWidth*nHeight*4;
    int size_bgr=nWidth*nHeight*3;
    int size_yuv=nWidth*nHeight*3/2;

    cudaMalloc((void**)&bgra_frames,sizeof(uint8_t)*size_bgra*n_stream*limit_);
    cudaMalloc((void**)&bgr_frames,sizeof(uint8_t)*size_bgr*n_stream*limit_);
    cudaMalloc((void**)&render_surfaces,sizeof(uint8_t)*size_bgra*n_stream*limit_);
    cudaMalloc((void**)&yuvs,sizeof(uint8_t)*size_yuv*n_stream*limit_);

    cudaMemset(bgra_frames,0,sizeof(uint8_t)*size_bgra*n_stream*limit_);
    cudaMemset(bgr_frames,0,sizeof(uint8_t)*size_bgr*n_stream*limit_);
    cudaMemset(render_surfaces,0,sizeof(uint8_t)*size_bgra*n_stream*limit_);
    cudaMemset(yuvs,0,sizeof(uint8_t)*size_yuv*n_stream*limit_);

    rect_ptr = (float*)malloc(sizeof(float) * 6* max_rect*limit_*n_stream);
    memset(rect_ptr, 0, sizeof(float) * 6* max_rect*limit_*n_stream);
     

    std::cerr<<"Finish Init!"<<std::endl;
    
    for(int i=0;i<n_stream;i++){
        std::thread t(decode_frame,
            std::ref(context),
            streams[i],
            dec_list[i],
            presenters[i],
            push_vec[i],
            bgra_frames,
            bgr_frames,
            render_surfaces,
            yuvs,
            end_arr,
            i
        );
        t.detach();
    }

    std::thread infer_thread(infer_trt,
        std::ref(context),
        streams,
        trt,
        bgr_frames
    );

    infer_thread.detach();

    int iter_wait=0;
    
    // while (iter_wait<100)
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // iter_wait++;
        // std::cerr<<"("<<iter_wait<<")wait for end!"<<std::endl;
        {
            std::unique_lock<std::mutex> uni_lock(lock_end);
            bool end_flag=true;
            for(int i=0;i<n_stream;i++){
                end_flag&=end_arr[i];
            }
            if(end_flag==true){
                end_infer=true;
                break;
            }
        }
    }

    cudaFree(bgra_frames);
    cudaFree(bgr_frames);
    cudaFree(render_surfaces);
    cudaFree(yuvs);
    free(rect_ptr);
    for(int i=0;i<n_stream;i++){
        // delete presenters[n_stream-1-i];
        delete dec_list[i];
        // delete engines[i];
    }
    delete presenters;
    delete trt;
    // free(presenters);
    free(dec_list);
    
    return;
}

std::vector<std::string> stringSplit(std::string &text,char sep){
    std::vector<std::string> tokens;
    std::istringstream tokenStream(text);
    std::string token;
    while(std::getline(tokenStream,token,sep)){
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc,char **argv){

    std::string inputUrl;
    std::string pushUrl;
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--inputUrl=", 11))
        {
            inputUrl = (argv[i] + 11);
        }
        else if (!strncmp(argv[i], "--pushUrl=", 10))
        {
            pushUrl = (argv[i] + 10);
        }
        else if (!strncmp(argv[i], "--isPush=", 8))
        {
            isPush = (argv[i] + 10);
        }
        
        else
        {
            std::cerr << "Invalid Argument: " << argv[i] << std::endl;
            return -1;
        }
    }

    std::vector<std::string> in_vec;
    std::vector<std::string> push_vec;

    in_vec=stringSplit(inputUrl,';');
    push_vec=stringSplit(pushUrl,';');
    if(in_vec.size()!=push_vec.size()){
        std::cout<<"ERROR!! input stream number not equal to output stream!!"<<std::endl;
        return -1;
    }
    for(int i=0;i<in_vec.size();i++){
        std::cout<<in_vec[i]<<" --> "<<push_vec[i]<<std::endl;
    }

    aysnc_process(in_vec,push_vec);


    return 0;
}