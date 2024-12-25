#include "yolov5_trt.h"
#include "Decode.h"
#include "bbox_calculate.h"

#include "Affine.h"
// #include "stream_push.h"


simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

Yolov5TRT::Yolov5TRT(int device_id){
    //device_id=1;
    //cudaSetDevice(device_id);
    //cudaStreamCreate(&stream);

    cudaGetDeviceProperties(&prop,device_id);
    printf("Compute capability : %d.%d\n", prop.major, prop.minor);

    // int nGpu=0;
    // cuDeviceGetCount(&nGpu);

    // printf("CUDA Deivce num: %d\n",nGpu);

    if (prop.major == 8 && prop.minor == 9) {
        engine_file = "../weights/Ada/yolov5s_fp16.trt";
    }
    else if(prop.major == 8 && prop.minor == 6) {
        engine_file = "../weights/Ampere/yolov5s_fp16.trt";
    }
    else if(prop.major == 7 && prop.minor == 5) {
        engine_file = "../weights/Turing/yolov5s_fp16.trt";
    }
    else if(prop.major == 6 && prop.minor == 1) {
        // engine_file = "../weights/Pascal/yolov5s_fp16.trt";
        // engine_file = "../weights/Pascal/yolov5s_qr_fp16.trt";//qr fp16
        // engine_file = "../weights/Pascal/yolov5s_qr_int8.trt";//qr int8
        // engine_file = "../weights/Pascal/yolov5_trimmed_qat_noqdq.INT8.trt";//qr int8
        engine_file = "../weights/Pascal/yolov5_trimmed_qat_noqdq.INT8.fixed.trt";//qr int8
    }


    pRect = (float*)malloc(sizeof(float) * 6* maxSize_rect);
    memset(pRect, 0, sizeof(float) * 6* maxSize_rect);
}

Yolov5TRT::~Yolov5TRT(){
    // cudaStreamDestroy(stream);
    cudaFreeHost(input_data_host);
    cudaFreeHost(output_data_host);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
}

void Yolov5TRT::init(int height,int width){

    // read the serialized engine 
    std::string trt_file(engine_file); 
    std::vector<char> trtModelStream_;
    size_t size(0);
    std::cout << "Loading engine file:" << trt_file << std::endl;
    std::ifstream file(trt_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        file.read(trtModelStream_.data(), size);
        file.close();
    } else {
        std::cerr << "Failed to open the engine file:" << trt_file << std::endl;
        return ;
    }

    runtime=nvinfer1::createInferRuntime(gLogger);
    assert(runtime);
    engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);

    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
 
    if(engine->getNbBindings() != 2){
        printf("你的onnx导出有问题,必须是1个输入和1个输出,你这明显有:%d个输出.\n", engine->getNbBindings() - 1);
        return;
    }

    // auto execution_context = make_nvshared(engine->createExecutionContext());
    execution_context=engine->createExecutionContext();

    input_dims = engine->getBindingDimensions(0);
    // input_dims.d[0] = input_batch;
    input_height = input_dims.d[2];
    input_width = input_dims.d[3];

    input_batch = 1;
    input_channel = 3;
    // input_height = 640;
    // input_width = 640;

    if(input_dims.d[0]<1){
        input_dims.d[0]=1;
    }

    input_numel = input_batch * input_channel * input_height * input_width;
    // float* input_data_host = nullptr;
    // float* input_data_device = nullptr;
    cudaMallocHost((void**)&input_data_host, input_numel * sizeof(float));
    cudaMalloc((void**)&input_data_device, input_numel * sizeof(float));


    // 3x3输入，对应3x3输出
    output_dims = engine->getBindingDimensions(1);
    output_numbox = output_dims.d[1];
    output_numprob = output_dims.d[2];

    if(output_dims.d[0]<1){
        output_dims.d[0]=1;
    }

    num_classes = output_numprob - 5;
    output_numel = input_batch * output_numbox * output_numprob;
    // float* output_data_host = nullptr;
    // float* output_data_device = nullptr;
    cudaMallocHost((void**)&output_data_host, sizeof(float) * output_numel);
    cudaMalloc((void**)&output_data_device, sizeof(float) * output_numel);

    cudaMalloc((void**)&bbox_device,sizeof(float)*(output_numbox*7+1));
    cudaMallocHost((void**)&bbox_host,sizeof(float)*(output_numbox*7+1));

    cudaMemset(bbox_device,0,sizeof(float)*(output_numbox*6+1));
    memset(bbox_host,0,sizeof(float)*(output_numbox*6+1));

    
    

    printf("InputDim:[%dx%dx%dx%d]\n",input_dims.d[0],input_dims.d[1],input_dims.d[2],input_dims.d[3]);
    printf("OutputDim:[%dx%dx%d]\n",output_dims.d[0],output_dims.d[1],output_dims.d[2]);

    std::cout << "num_classes-->" << num_classes << std::endl;
    //std::cout << "runtime-->" << runtime << std::endl;
    //std::cout << "engine-->" << engine << std::endl;
    //std::cout << "execution_context-->" << execution_context << std::endl;
    // exit(0);

    // cv::Mat image;
    // cv::VideoCapture cap;
    // cap.open(url);
    // cap.read(image);

    // auto image = cv::imread("car.jpg");
    // 通过双线性插值对图像进行resize
    float scale_x = input_width / (float)width;
    float scale_y = input_height / (float)height;
    float scale = (std::min)(scale_x, scale_y);
    // float i2d[6], d2i[6];
    // resize图像，源图像和目标图像几何中心的对齐



    cudaMallocHost((void**)&i2d,sizeof(float)*6);
    cudaMallocHost((void**)&d2i,sizeof(float)*6);
    i2d[0] = scale;  
    i2d[1] = 0;  
    i2d[2] = (-scale * width + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  
    i2d[4] = scale;  
    i2d[5] = (-scale * height + input_height + scale - 1) * 0.5;
 
    // cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
    // cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
    m2x3_i2d=cv::Mat(2, 3, CV_32F, i2d);
    m2x3_d2i=cv::Mat(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    // cv::Mat input_image(input_height, input_width, CV_8UC3);
    input_image=cv::Mat(input_height, input_width, CV_8UC3);

    image.create(720, 1280, CV_8UC3);

    class_num = cocolabels.size();
    pColors = (float*)malloc(sizeof(float) * 3 *class_num);
    memset(pColors, 0, sizeof(float) * 3 * class_num);

    cudaMalloc((void**)&d2i_d,sizeof(float)*6);
    cudaMemcpy(d2i_d,d2i,sizeof(float)*6,cudaMemcpyHostToDevice);

    //ColorInitForD3D11

    for (int k = 0; k < cocolabels.size(); k++) {
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(k);
        //std::tuple<uint8_t, uint8_t, uint8_t> color = random_color(k);
        //pColors[3 * k + 0] = std::get<0>(color);
        //pColors[3 * k + 1] = std::get<1>(color);
        pColors[3 * k + 0] = color[2]/ 255.f;
        pColors[3 * k + 1] = color[1]/ 255.f;
        pColors[3 * k + 2] = color[0]/ 255.f;
        //std::cout << color[0] << "," << color[1] << "," << color[2] << std::endl;
        //std::cout << pColors[3 * k + 0] << "," << pColors[3 * k + 1] << "," << pColors[3 * k + 2] << std::endl;
    }

    
}

void Yolov5TRT::inference_cv(cudaStream_t &stream){
//void Yolov5TRT::inference_cv(){

    //int device_id = 1;
    //cudaSetDevice(device_id);
    //cudaStream_t stream = nullptr;
    //cudaStreamCreate(&stream);

    // // cv::Mat ori_img;
    // cv::Mat image;
    // cv::VideoCapture cap;
    // // auto fourcc=cv::VideoWriter::fourcc('X','V','I','D');
    // // cv::VideoWriter vw("result.avi",fourcc,10,cv::Size(640,480),1);
    


    // // std::string url1="rtsp://admin:ld123456@192.168.200.112:554";
    //std::string url1="06_720p.mp4";
    // //std::string url1="F:/maomao.mp4";
    // // std::string url1="/media/dl/Dasego/YOLOv7_OpenVINO_cpp-python/data/Tag/2.mp4";
    //cap.open(url);

    cap.read(image);

    std::cout<<"inference!"<<std::endl;
    
    double start, end, res;
    double total_s, total_e, total_res;
    total_s = cv::getTickCount();
    
    int count = 0;

    std::string str1;
    std::string str0;
    std::string str2;

    std::cout << "runtime1-->" << runtime << std::endl;
    std::cout << "engine1-->" << engine << std::endl;
    std::cout << "execution_context1-->" << execution_context << std::endl;

    while (count < 1200) {
        ++count;


        str0 = "PrepareImage-" + std::to_string(count);
        nvtxRangePushA(str0.c_str());
        cap.read(image);



        //start = cv::getTickCount();



        cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆
        // cv::imwrite("input-image.jpg", input_image);

        //std::cout << "sum(input_image)-->" << cv::sum(input_image) << std::endl;

        int image_area = input_image.cols * input_image.rows;
        unsigned char* pimage = input_image.data;
        float* phost_b = input_data_host + image_area * 0;
        float* phost_g = input_data_host + image_area * 1;
        float* phost_r = input_data_host + image_area * 2;
        for (int i = 0; i < image_area; ++i, pimage += 3) {
            // 注意这里的顺序rgb调换了
            *phost_r++ = pimage[0] / 255.0f;
            *phost_g++ = pimage[1] / 255.0f;
            *phost_b++ = pimage[2] / 255.0f;
        }
        ///

        nvtxRangePop();

        start = cv::getTickCount();


        str1 = "Infer-" + std::to_string(count);
        nvtxRangePushA(str1.c_str());

        cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);





        execution_context->setBindingDimensions(0, input_dims);
        float* bindings[] = { input_data_device, output_data_device };

        bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);

        cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream);
        //cudaStreamSynchronize(stream);
        end = cv::getTickCount();

        nvtxRangePop();

        str2 = "PostProcess-" + std::to_string(count);
        nvtxRangePushA(str2.c_str());
        std::vector<std::vector<float>> bboxes;
        float confidence_threshold = 0.25;
        float nms_threshold = 0.5;
        for (int i = 0; i < output_numbox; ++i) {
            float* ptr = output_data_host + i * output_numprob;
            float objness = ptr[4];
            if (objness < confidence_threshold)
                continue;

            float* pclass = ptr + 5;
            int label = std::max_element(pclass, pclass + num_classes) - pclass;
            float prob = pclass[label];
            float confidence = prob * objness;
            if (confidence < confidence_threshold)
                continue;


            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;


            float image_base_left = d2i[0] * left + d2i[2];
            float image_base_right = d2i[0] * right + d2i[2];
            float image_base_top = d2i[0] * top + d2i[5];
            float image_base_bottom = d2i[0] * bottom + d2i[5];

            bboxes.push_back({ image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence });
        }
         //printf("decoded bboxes.size = %d\n", bboxes.size());


        std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b) {return a[5] > b[5]; });
        std::vector<bool> remove_flags(bboxes.size());
        std::vector<std::vector<float>> box_result;
        box_result.reserve(bboxes.size());

        auto iou = [](const std::vector<float>& a, const std::vector<float>& b) {
            float cross_left = (std::max)(a[0], b[0]);
            float cross_top = (std::max)(a[1], b[1]);
            float cross_right = (std::min)(a[2], b[2]);
            float cross_bottom = (std::min)(a[3], b[3]);

            float cross_area = (std::max)(0.0f, cross_right - cross_left) * (std::max)(0.0f, cross_bottom - cross_top);
            float union_area = (std::max)(0.0f, a[2] - a[0]) * (std::max)(0.0f, a[3] - a[1])
                + (std::max)(0.0f, b[2] - b[0]) * (std::max)(0.0f, b[3] - b[1]) - cross_area;
            if (cross_area == 0 || union_area == 0) return 0.0f;
            return cross_area / union_area;
        };

        for (int i = 0; i < bboxes.size(); ++i) {
            if (remove_flags[i]) continue;

            auto& ibox = bboxes[i];
            box_result.emplace_back(ibox);
            for (int j = i + 1; j < bboxes.size(); ++j) {
                if (remove_flags[j]) continue;

                auto& jbox = bboxes[j];
                if (ibox[4] == jbox[4]) {
                    // class matched
                    if (iou(ibox, jbox) >= nms_threshold)
                        remove_flags[j] = true;
                }
            }
        }



        // printf("box_result.size = %d\n", box_result.size());
        //end = cv::getTickCount();

        for (int i = 0; i < box_result.size(); ++i) {
            auto& ibox = box_result[i];
            float left = ibox[0];
            float top = ibox[1];
            float right = ibox[2];
            float bottom = ibox[3];
            int class_label = ibox[4];
            float confidence = ibox[5];
            //std::cout << "(" << ibox[0] << "," << ibox[1] << "," << ibox[2] << "," << ibox[3] << ")" << std::endl;
            cv::Scalar color;
            std::tie(color[0], color[1], color[2]) = random_color(class_label);
            cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 2);

            // auto name      = cocolabels[class_label];
            // auto caption   = cv::format("%s %.2f", name, confidence);
            // int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

            std::string name = cocolabels[class_label];
            // auto caption   = cv::format("%s %.2f", name, confidence);
            std::string caption = name + " " + std::to_string(confidence);
            // std::cout<<"Class--> "<<caption<<std::endl;
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(left - 3, top - 33), cv::Point(left + text_width, top), color, -1);
            cv::putText(image, caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 1, 16);
        }
        nvtxRangePop();
        res = (end - start) / cv::getTickFrequency();
        std::cout << "time of output --> " << res * 1000 << " ms" << " iter-> " << count << std::endl;
        std::string time_cost = std::to_string(res * 1000) + " ms";
        cv::putText(image, time_cost, cv::Point(200, 20), 0, 1, cv::Scalar(255, 255, 255), 2, 8);

        cv::imshow("rst", image);
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }
    total_e = cv::getTickCount();
    total_res = (total_e - total_s) / cv::getTickFrequency();
    std::cout << "++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "time of Total --> " << total_res * 1000 << " ms" << std::endl;
    std::cout << "time of Avg --> " << total_res * 1000/(count+1) << " ms" << std::endl;

    // cudaStreamDestroy(stream);
    // cudaFreeHost(input_data_host);
    // cudaFreeHost(output_data_host);
    // cudaFree(input_data_device);
    // cudaFree(output_data_device);
}


void Yolov5TRT::inference(
    // cv::Mat &image,
    uint8_t *image_d,
    cudaStream_t &stream){

    
    double start, end, res;
    double total_s, total_e, total_res;
    total_s = cv::getTickCount();
    

    std::string str1;
    std::string str0;
    std::string str2;

    int src_width=1280;
    int src_height=720;
        
    

    // str0 = "PrepareImage-" + std::to_string(count);
    // nvtxRangePushA(str0.c_str());
    // cap.read(image);
    // std::cout<<"image: ["<<image.rows<<","<<image.cols<<"]"<<std::endl;
    
    //cv::Mat image;
    //cudaMemcpyAsync(image.data, image_d, sizeof(uint8_t) * 3 * 1280 * 720,cudaMemcpyDeviceToHost,stream);

    //cv::imshow("src", image);
    //cv::waitKey(16);
    //start = cv::getTickCount();

    /*
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆
    // cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
    ///

    // cv::imshow("ori",image);
    // // cv::imshow("ori1",input_image);
    // char key1=cv::waitKey(10);
    // std::cout<<image.rows<<","<<image.cols<<std::endl;

    // nvtxRangePop();

    start = cv::getTickCount();


    // str1 = "Infer-" + std::to_string(count);
    // nvtxRangePushA(str1.c_str());

    cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);

    */

    warp_affine_bilinear(image_d,src_width*3,src_width,src_height,
        input_data_device,input_width*3,input_width,input_height,114,stream
    );

    
    // cv::imshow("test",test);
    // char key=cv::waitKey(16);


    execution_context->setBindingDimensions(0, input_dims);

    float* bindings[] = {input_data_device, output_data_device};

    bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);

    // cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream);
    
    cudaMemsetAsync(bbox_device,0,sizeof(float)*(output_numbox*7+1),stream);
    // nvtxRangePop();

    // str2 = "PostProcess-" + std::to_string(count);
    // nvtxRangePushA(str2.c_str());
    std::vector<std::vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;


    // for(int i = 0; i < output_numbox; ++i){
    //     float* ptr = output_data_host + i * output_numprob;
    //     float objness = ptr[4];
    //     if(objness < confidence_threshold)
    //         continue;

    //     float* pclass = ptr + 5;
    //     int label     = std::max_element(pclass, pclass + num_classes) - pclass;
    //     float prob    = pclass[label];
    //     float confidence = prob * objness;
    //     if(confidence < confidence_threshold)
    //         continue;

        
    //     float cx     = ptr[0];
    //     float cy     = ptr[1];
    //     float width  = ptr[2];
    //     float height = ptr[3];

        
    //     float left   = cx - width * 0.5;
    //     float top    = cy - height * 0.5;
    //     float right  = cx + width * 0.5;
    //     float bottom = cy + height * 0.5;

        
    //     float image_base_left   = d2i[0] * left   + d2i[2];
    //     float image_base_right  = d2i[0] * right  + d2i[2];
    //     float image_base_top    = d2i[0] * top    + d2i[5];
    //     float image_base_bottom = d2i[0] * bottom + d2i[5];
    //     bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
    // }

    //gpu kernel
    dim3 threadSize(32);
    dim3 blockSize((output_numbox+31)/32);
    bbox_calculate(output_data_device,bbox_device,d2i_d,
        output_numbox,output_numprob,
        num_classes,confidence_threshold,stream);

    cudaMemcpyAsync(bbox_host,bbox_device,sizeof(float)*(output_numbox*7+1),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    end = cv::getTickCount();

    // printf("old bboxes.size = %d\n", bboxes.size());
    // printf("new bboxes.size = %f\n", bbox_host[0]);
    // printf("output_numbox = %d output_numprob = %d\n", output_numbox,output_numprob);
    // std::cerr<<"["<<bbox_host[1+6]<<","<<bbox_host[2+6]<<","<<bbox_host[3+6]<<
    //     ","<<bbox_host[4+6]<<","<<bbox_host[5+6]<<","<<bbox_host[6+6]<<"]"<<std::endl;
    
    int num_bbox=(int)bbox_host[0];
    for(int k=0;k<output_numbox;k++){
        if(bbox_host[k*7+7]>0){
            bboxes.push_back({
                bbox_host[k*7+1], bbox_host[k*7+2], 
                bbox_host[k*7+3], bbox_host[k*7+4], 
                bbox_host[k*7+5], bbox_host[k*7+6]});

            // std::cerr<<"["<<bbox_host[k*7+1]<<","<<bbox_host[k*7+2]<<","<<bbox_host[k*7+3]<<
            // ","<<bbox_host[k*7+4]<<","<<bbox_host[k*7+5]<<","<<bbox_host[k*7+6]<<"]"<<std::endl;
        }
        
    }

    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b){return a[5] > b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a, const std::vector<float>& b){
        float cross_left   = (std::max)(a[0], b[0]);
        float cross_top    = (std::max)(a[1], b[1]);
        float cross_right  = (std::min)(a[2], b[2]);
        float cross_bottom = (std::min)(a[3], b[3]);

        float cross_area = (std::max)(0.0f, cross_right - cross_left) * (std::max)(0.0f, cross_bottom - cross_top);
        float union_area = (std::max)(0.0f, a[2] - a[0]) * (std::max)(0.0f, a[3] - a[1]) 
                        + (std::max)(0.0f, b[2] - b[0]) * (std::max)(0.0f, b[3] - b[1]) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[4] == jbox[4]){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }

    p_num = box_result.size();
    for (int i = 0; i < box_result.size(); ++i) {
        pRect[6 * i + 0] = box_result[i][0];
        pRect[6 * i + 1] = box_result[i][1];
        pRect[6 * i + 2] = box_result[i][2];
        pRect[6 * i + 3] = box_result[i][3];
        pRect[6 * i + 4] = box_result[i][4];
        pRect[6 * i + 5] = box_result[i][5];
        // std::cout << "pRect(" << pRect[6 * i] << "," << pRect[6 * i + 1] << "," << pRect[6 * i + 2] << "," << pRect[6 * i + 3] <<")"<< std::endl;
        // std::cout << "box_result(" << box_result[i][0] << "," << box_result[i][1] << "," << box_result[i][2] << "," << box_result[i][3] <<")"<< std::endl;
    }

    //printf("box_result.size = %d\n", box_result.size());
    //end = cv::getTickCount();

    /*
    cv::Mat image;
    image.create(720,1280,CV_8UC3);
    cudaMemcpyAsync(image.data, image_d, sizeof(uint8_t) * 3 * 720 * 1280,cudaMemcpyDeviceToHost,stream);
    
    for(int i = 0; i < box_result.size(); ++i){
        auto& ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        //std::cout << "ibox(" << ibox[0] << "," << ibox[1] << "," << ibox[2] << "," << ibox[3] << ")" << std::endl;
        //std::cout << "(" << left << "," << top << "," << right << "," << bottom << ")" << std::endl;
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 2);

        // auto name      = cocolabels[class_label];
        // auto caption   = cv::format("%s %.2f", name, confidence);
        // int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

        std::string name      = cocolabels[class_label];
        // auto caption   = cv::format("%s %.2f", name, confidence);
        std::string caption=name+" "+std::to_string(confidence);
         //std::cout<<"Class--> "<<caption<<std::endl;
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 1, 16);
    }

    // nvtxRangePop();
    res = (end - start) / cv::getTickFrequency();
    // std::cout << "time of output --> " << res*1000<<" ms"<<" iter-> "<<count<<std::endl;
    std::string time_cost=std::to_string(res*1000)+" ms";
    cv::putText(image, time_cost, cv::Point(200, 20), 0, 1, cv::Scalar(255,255,255), 2, 8);
    cv::imshow("rst",image);
    char key=cv::waitKey(1);
    if(key=='q'){
       return;
    }
    
    // total_e = cv::getTickCount();
    // total_res = (total_e - total_s) / cv::getTickFrequency();
    // std::cout << "++++++++++++++++++++++++++++++" << std::endl;
    // std::cout << "time of Total --> " << total_res * 1000 << " ms" << std::endl;
    // std::cout << "time of Avg --> " << total_res * 1000/(count+1) << " ms" << std::endl;

    */

}



