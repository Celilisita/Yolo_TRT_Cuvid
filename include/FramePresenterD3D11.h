/*
* Copyright 2017-2023 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#pragma once

#include <iostream>
#include <mutex>
#include <thread>
#include <d3d11.h>
#include <d2d1.h>
#include <dwrite.h>
#include <cuda.h>
#include <cudaD3D11.h>
#include "FramePresenterD3D.h"
#include "../Utils/NvCodecUtils.h"

#include <Windows.h>
#include <wrl/client.h>

/**
* @brief D3D11 presenter class derived from FramePresenterD3D
*/
class FramePresenterD3D11 : public FramePresenterD3D
{
public:
    /**
    *   @brief  FramePresenterD3D11 constructor. This will launch a rendering thread which will be fed with decoded frames
    *   @param  cuContext - CUDA context handle
    *   @param  nWidth - Width of D3D surface
    *   @param  nHeight - Height of D3D surface
    */
    FramePresenterD3D11(CUcontext cuContext, int nWidth, int nHeight, char* winApp, CUstream stream) :
        FramePresenterD3D(cuContext, nWidth, nHeight,winApp,stream) 
    {
        pthMsgLoop = new std::thread(ThreadProc, this);
        while (!bReady) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // hTimerQueue = CreateTimerQueue();
        hPresentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

        p_num = 0;
        pRect = (float*)malloc(sizeof(float) * 6* max_size);
        memset(pRect, 0.0f, 6 * max_size);
        // cudaMalloc((void**)&render_surface,sizeof(uint8_t)*4*nWidth*nHeight);
        // frame_bgra.create(720,1280,CV_8UC4);
        // size_bgra=720*1280*4;

        //D2DInit();
    }

    /**
    *   @brief  FramePresenterD3D11 destructor.
    */
    ~FramePresenterD3D11() {
        printf("~Free D3D11 Presenter~\n");
        // if (hTimerQueue)
        // {
        //     DeleteTimerQueue(hTimerQueue);
        // }
        if (hPresentEvent)
        {
            CloseHandle(hPresentEvent);
        }
        bQuit = true;
        pthMsgLoop->join();
        delete pthMsgLoop;

        
    }

    /**
    *   @brief  Presents a frame present in host memory. More specifically, it copies the host surface
    *           data to a d3d staging texture and then copies it to the swap chain backbuffer for presentation
    *   @param  pData - pointer to host surface data
    *   @param  nBytes - number of bytes to copy
    *   @return true on success
    *   @return false when the windowing thread is not ready to be served
    */
    bool PresentHostFrame(BYTE *pData, int nBytes) {
        mtx.lock();
        if (!bReady) {
            mtx.unlock();
            return false;
        }

        D3D11_MAPPED_SUBRESOURCE mappedTexture;
        ck(pContext->Map(pStagingTexture, 0, D3D11_MAP_WRITE, 0, &mappedTexture));
        memcpy(mappedTexture.pData, pData, min(nWidth * nHeight * 4, nBytes));
        pContext->Unmap(pStagingTexture, 0);
        pContext->CopyResource(pBackBuffer, pStagingTexture);
        ck(pSwapChain->Present(0, 0));
        mtx.unlock();
        return true;
    }

    std::wstring string2wstring(std::string &s) {

        std::string strLocale = setlocale(LC_ALL, "");
        const char* chSrc = s.c_str();
        size_t nDestSize = mbstowcs(NULL, chSrc, 0) + 1;
        wchar_t* wchDest = new wchar_t[nDestSize];
        wmemset(wchDest, 0, nDestSize);
        mbstowcs(wchDest, chSrc, nDestSize);
        std::wstring wstrResult = wchDest;
        delete[] wchDest;
        setlocale(LC_ALL, strLocale.c_str());
        return wstrResult;
    }

    std::string floatToString(float value, int precision) {
        std::ostringstream streamObj;
        // 设置小数点后的位数
        streamObj << std::fixed << std::setprecision(precision) << value;
        return streamObj.str();
    }

    void setWinName(std::string name){
        winName=name;
    }

    void drawRectText() {


        //rc.left = rand() % 255;
        //rc.top = rand() % 255;
        //rc.right = rand() % 255 + 200;
        //rc.bottom = rand() % 255 + 300;
        //std::cout << "m_pd2dRenderTarget-->" << m_pd2dRenderTarget << std::endl;
        std::wstring textStr;
        if (m_pd2dRenderTarget != nullptr)
        {
            m_pd2dRenderTarget->BeginDraw();
            

            for (int k = 0; k < p_num; k++) {
                rc.left = pRect[6 * k];
                rc.top = pRect[6 * k + 1];
                rc.right = pRect[6 * k + 2];
                rc.bottom = pRect[6 * k + 3];

                int class_idx=int(pRect[6 * k + 4]);

                std::string score=floatToString(pRect[6*k+5],3);

                m_pd2dRenderTarget->DrawRectangle(
                    rc,
                    //m_pColorBrush.Get(),
                    pColorBrushs[class_idx],
                    3.0f
                );
                textStr = wlabels[class_idx]+L" "+string2wstring(score);
                m_pd2dRenderTarget->DrawTextA(
                    textStr.c_str(), 
                    (UINT32)textStr.size(), 
                    m_pTextFormat.Get(),
                    D2D1_RECT_F{ rc.left, rc.top-25.f, rc.left+200.f, rc.top+100.0f }, 
                    //rc,
                    pColorBrushs[class_idx]
                );
            }
            
            m_pd2dRenderTarget->EndDraw();

            //std::cout << "Rc: " << rc.left << std::endl;

        }
    }

    bool PresentDeviceFrame(unsigned char *dpBgra, unsigned char *render_result,int nPitch, int64_t delay) {
        //mtx.lock();
        //if (!bReady) {
        //    mtx.unlock();
        //    return false;
        //}
        //// cudaMemcpy(frame_bgra.data,dpBgra,sizeof(uint8_t)*size_bgra,cudaMemcpyDeviceToHost);
        //// cv::imshow("frame", frame_bgra);
        //// char key = cv::waitKey(16);
        //// if (key == 'q') {
        ////     exit(0);
        //// }
        
        CopyDeviceFrame(dpBgra, nPitch);

        drawRectText();

        Direct2Cuda(render_result,nPitch);

        pSwapChain->Present(0,0);

        //if (!CreateTimerQueueTimer(&hTimer, hTimerQueue,
        //    (WAITORTIMERCALLBACK)PresentRoutine, this, (DWORD)delay, 0, 0))
        //{
        //    std::cout << "Problem in createtimer" << std::endl;
        //}
        //while (WaitForSingleObject(hPresentEvent, 0) != WAIT_OBJECT_0)
        //{
        //}
        //if (hTimer)
        //{
        //    DeleteTimerQueueTimer(hTimerQueue, hTimer, nullptr);
        //}
        //mtx.unlock();

        

        return true;
    }

    void D2DInit() {
        // std::cout << "D2Dinit-------------" << std::endl;
        HRESULT hr1 = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, m_pd2dFactory.GetAddressOf());
        HRESULT hr2 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
            reinterpret_cast<IUnknown**>(m_pdwriteFactory.GetAddressOf()));

        m_pd2dRenderTarget.Reset();

        // std::cout << "m_pd2dRenderTarget.GetAddressOf()-->" << m_pd2dRenderTarget.GetAddressOf() << std::endl;
       
        HRESULT hr3 = pSwapChain->GetBuffer(0, __uuidof(IDXGISurface), reinterpret_cast<void**>(surface.GetAddressOf()));
        D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(
            D2D1_RENDER_TARGET_TYPE_DEFAULT,
            D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED));

        //HRESULT hr = m_pd2dFactory->CreateDxgiSurfaceRenderTarget(surface.Get(), &props, m_pd2dRenderTarget.GetAddressOf());
        HRESULT hr = m_pd2dFactory->CreateDxgiSurfaceRenderTarget(surface.Get(), &props, &m_pd2dRenderTarget);

        

        // std::cout << "hr1->" << hr1 << std::endl;
        // std::cout << "hr2->" << hr2 << std::endl;
        // std::cout << "hr3->" << hr3 << std::endl;
        // std::cout << "hr->" << hr << std::endl;
        // std::cout << "pSwapChain->" << pSwapChain << std::endl;
        // std::cout << "surface->" << surface << std::endl;
        // std::cout << "m_pd2dFactory->" << m_pd2dFactory << std::endl;
        
        // std::cout << "m_pd2dRenderTarget-->" << m_pd2dRenderTarget<< std::endl;

        surface.Reset();

        if (hr == E_NOINTERFACE)
        {
            std::cout << "E_NOINTERFACE" << std::endl;
        }
        else if (hr == S_OK)
        {
            HRESULT hr5 = m_pd2dRenderTarget->CreateSolidColorBrush(
                D2D1::ColorF(D2D1::ColorF::White),
                m_pColorBrush.GetAddressOf());

            HRESULT hr6 = m_pdwriteFactory->CreateTextFormat(L"宋体", nullptr, DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 20, L"zh-cn",
                m_pTextFormat.GetAddressOf());

            // std::cout << "hr5->" << hr5 << std::endl;
            // std::cout << "hr6->" << hr6 << std::endl;
        }
        else
        {
            std::cout << "ELSE Error!" << std::endl;

            assert(m_pd2dRenderTarget);
        }


    }

    bool PresentRectFrame(float* rectArray, int number) {

        //memset(pRect, 0.0f, 1000);
        p_num = number;
        memcpy(pRect, rectArray, sizeof(float) * 6 * number);

        /*std::cout << "number->" << number << std::endl;
        for (int i = 0; i < number; i++) {
            
            std::cout << "(" << pRect[4 * i] << "," << pRect[4 * i + 1] << "," << pRect[4 * i + 2] << "," << pRect[4 * i + 3] <<")"<< std::endl;
        }*/

        return true;
    }

    void colorInit(float* colors, int num_class) {
        class_num = num_class;
        pColors = (float*)malloc(sizeof(float) * 3 * class_num);
        memcpy(pColors, colors, sizeof(float) * 3 * class_num);

        pColorBrushs = (ID2D1SolidColorBrush**)malloc(sizeof(ID2D1SolidColorBrush*) * class_num);

        for (int i = 0; i < class_num; i++) {

            HRESULT hr = m_pd2dRenderTarget->CreateSolidColorBrush(
                D2D1::ColorF(pColors[3*i+0], pColors[3 * i+1], pColors[3 * i+2],1.0f),
                &pColorBrushs[i]);

            //std::cout << pColors[3 * i + 0] << "," << pColors[3 * i + 1] << "," << pColors[3 * i + 2] << std::endl;
        }

        for (int k = 0; k < class_num; k++) {
            std::wstring wstr = string2wstring(labels[k]);
            wlabels.emplace_back(wstr);
        }
    }

private:
    /**
    *   @brief  Launches the windowing functionality
    *   @param  This - pointer to FramePresenterD3D11 object
    */
    static void ThreadProc(FramePresenterD3D11 *This) {
        This->Run();
    }
    /**
    *   @brief  Callback called by D3D runtime. This callback is registered during call to
    *           CreateTimerQueueTimer in PresentDeviceFrame. In CreateTimerQueueTimer we also
    *           set a timer. When this timer expires this callback is called. This functionality
    *           is present to facilitate timestamp based presentation.
    *   @param  lpParam - void pointer to client data
    *   @param  TimerOrWaitFired - TRUE for this callback as this is a Timer based callback (Refer:https://docs.microsoft.com/en-us/previous-versions/windows/desktop/legacy/ms687066(v=vs.85))
    */
    static VOID CALLBACK PresentRoutine(PVOID lpParam, BOOLEAN TimerOrWaitFired)
    {
        if (!lpParam) return;
        FramePresenterD3D11* presenter = (FramePresenterD3D11 *)lpParam;
        presenter->pSwapChain->Present(1, 0);
        SetEvent(presenter->hPresentEvent);
    }

    /**
    *   @brief This function is on a thread spawned during FramePresenterD3D11 construction.
    *          It creates the D3D window and monitors window messages in a loop. This function
    *          also creates swap chain for presentation and also registers the swap chain backbuffer
    *          with cuda.
    */
    void Run() {
        // HWND hwndMain = CreateAndShowWindow(nWidth, nHeight,"D3D11");
        char *win_name=new char[winName.length()+1];
        strcpy(win_name,winName.c_str());
        HWND hwndMain = CreateAndShowWindow(nWidth, nHeight,win_name);

        std::cout << "nWidth: " << nWidth << " nHeight: " << nHeight << std::endl;

        DXGI_SWAP_CHAIN_DESC sc = { 0 };
        sc.BufferCount = 1;
        sc.BufferDesc.Width = nWidth;
        sc.BufferDesc.Height = nHeight;
        sc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sc.BufferDesc.RefreshRate.Numerator = 0;
        sc.BufferDesc.RefreshRate.Denominator = 1;
        sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sc.OutputWindow = hwndMain;
        sc.SampleDesc.Count = 1;
        sc.SampleDesc.Quality = 0;
        sc.Windowed = TRUE;

        //ID3D11Device* pDevice = NULL;
        ComPtr<ID3D11Device> pDevice = nullptr;

        
        HRESULT hr = S_OK;
        UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;	// Direct2D

        //D3D_DRIVER_TYPE driverTypes[] =
        //{
        //    D3D_DRIVER_TYPE_HARDWARE,
        //    D3D_DRIVER_TYPE_WARP,
        //    D3D_DRIVER_TYPE_REFERENCE,
        //};
        D3D_DRIVER_TYPE driverTypes[] =
        {
            D3D_DRIVER_TYPE_UNKNOWN ,
            D3D_DRIVER_TYPE_HARDWARE ,
            D3D_DRIVER_TYPE_REFERENCE ,
            D3D_DRIVER_TYPE_NULL ,
            D3D_DRIVER_TYPE_SOFTWARE ,
            D3D_DRIVER_TYPE_WARP
        };
        UINT numDriverTypes = ARRAYSIZE(driverTypes);


        D3D_FEATURE_LEVEL featureLevels[] =
        {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };
        UINT numFeatureLevels = ARRAYSIZE(featureLevels);
        D3D_FEATURE_LEVEL featureLevel;
        D3D_DRIVER_TYPE d3dDriverType;

        for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++)
        {
            d3dDriverType = driverTypes[driverTypeIndex];
            hr = D3D11CreateDevice(GetAdapterByContext(cuContext), d3dDriverType, nullptr, createDeviceFlags, featureLevels, numFeatureLevels,
                D3D11_SDK_VERSION, &pDevice, &featureLevel, m_pd3dImmediateContext.GetAddressOf());
            // std::cout << "D3D11CreateDevice->" << hr << std::endl;
            if (hr == E_INVALIDARG)
            {
                std::cout << "D3D11CreateDevice-->E_INVALIDARG" << std::endl;
                
                hr = D3D11CreateDevice(GetAdapterByContext(cuContext), d3dDriverType, nullptr, createDeviceFlags, &featureLevels[1], numFeatureLevels - 1,
                    D3D11_SDK_VERSION, &pDevice, &featureLevel, m_pd3dImmediateContext.GetAddressOf());
            }

            if (SUCCEEDED(hr))
                break;
        }

        


        // UINT m_4xMsaaQuality;
        // bool m_Enable4xMsaa = false;
        // hr=pDevice->CheckMultisampleQualityLevels(
        //     DXGI_FORMAT_B8G8R8A8_UNORM, 4, &m_4xMsaaQuality);	
        // assert(m_4xMsaaQuality > 0);
        // std::cout << "CheckMultisampleQualityLevels->" << hr << std::endl;

        ComPtr<IDXGIDevice> dxgiDevice = nullptr;
        ComPtr<IDXGIAdapter> dxgiAdapter = nullptr;
        ComPtr<IDXGIFactory1> dxgiFactory1 = nullptr;	// 

      
        // "IDXGIFactory::CreateSwapChain: This function is being called with a device from a different IDXGIFactory."
        //hr=pDevice->As(&dxgiDevice);
        hr = pDevice.As(&dxgiDevice);
        // std::cout << "As->" << hr << std::endl;
        hr = dxgiDevice->GetAdapter(dxgiAdapter.GetAddressOf());
        // std::cout << "GetAdapter-->" <<hr<< std::endl;
        hr = dxgiAdapter->GetParent(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(dxgiFactory1.GetAddressOf()));
        // std::cout << "GetParent-->" <<hr<< std::endl;


        // if (m_Enable4xMsaa)
        // {
        //     sc.SampleDesc.Count = 4;
        //     sc.SampleDesc.Quality = m_4xMsaaQuality - 1;
        // }
        // else
        // {
        //     sc.SampleDesc.Count = 1;
        //     sc.SampleDesc.Quality = 0;
        // }
        // sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        // sc.BufferCount = 1;
        // sc.OutputWindow = hwndMain;
        // sc.Windowed = TRUE;
        // sc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        // sc.Flags = 0;
        hr=dxgiFactory1->CreateSwapChain(pDevice.Get(), &sc, &pSwapChain);
        //D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_UNKNOWN,
            //NULL, 0, NULL, 0, D3D11_SDK_VERSION, &sd, m_pSwapChain.GetAddressOf(), &m_pd3dDevice, NULL, NULL);
        // std::cout << "CreateSwapChain-->" << hr << std::endl;

        //exit(0);

        

        //ck(D3D11CreateDeviceAndSwapChain(GetAdapterByContext(cuContext), D3D_DRIVER_TYPE_UNKNOWN,
        //    NULL, 0, NULL, 0, D3D11_SDK_VERSION, &sc, &pSwapChain, &pDevice, NULL, &pContext));


        ck(pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer));

        D2DInit();

        // D3D11_TEXTURE2D_DESC td;
        // pBackBuffer->GetDesc(&td);
        // td.BindFlags = 0;
        // td.Usage = D3D11_USAGE_STAGING;
        // td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        //ck(pDevice->CreateTexture2D(&td, NULL, &pStagingTexture));

        ck(cuCtxPushCurrent(cuContext));
        ck(cuGraphicsD3D11RegisterResource(&cuResource, pBackBuffer, CU_GRAPHICS_REGISTER_FLAGS_NONE));
        ck(cuGraphicsResourceSetMapFlags(cuResource, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
        ck(cuCtxPopCurrent(NULL));

        bReady = true;
        MSG msg = { 0 };
        while (!bQuit && msg.message != WM_QUIT) {
            if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }

        mtx.lock();
        bReady = false;
        ck(cuCtxPushCurrent(cuContext));
        ck(cuGraphicsUnregisterResource(cuResource));
        ck(cuCtxPopCurrent(NULL));


        pStagingTexture->Release();
        pBackBuffer->Release();
        pContext->Release();

        m_pdwriteFactory->Release();
        m_pd2dRenderTarget->Release();
        m_pd2dFactory->Release();
        surface->Release();
        dxgiFactory1->Release();
        dxgiAdapter->Release();
        dxgiDevice->Release();

        pDevice->Release();
        pSwapChain->Release();
        DestroyWindow(hwndMain);
        mtx.unlock();
    }

    /**
    *   @brief  Gets the DXGI adapter on which the given cuda context is current
    *   @param   CUcontext - handle to cuda context
    *   @return  pAdapter - pointer to DXGI adapter
    *   @return  NULL - In case there is no adapter corresponding to the supplied cuda context
    */
    static IDXGIAdapter *GetAdapterByContext(CUcontext cuContext) {
        CUdevice cuDeviceTarget;
        ck(cuCtxPushCurrent(cuContext));
        ck(cuCtxGetDevice(&cuDeviceTarget));
        ck(cuCtxPopCurrent(NULL));

        char szDeviceName[80];
        cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDeviceTarget);
        std::cout << "FramePresenter Display : GPU in use: " << szDeviceName << std::endl;

        IDXGIFactory1 *pFactory = NULL;
        ck(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory));
        IDXGIAdapter *pAdapter = NULL;
        for (unsigned i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; i++) {
            CUdevice cuDevice;
            ck(cuD3D11GetDevice(&cuDevice, pAdapter));
            if (cuDevice == cuDeviceTarget) {
                pFactory->Release();
                DXGI_ADAPTER_DESC desc;
                pAdapter->GetDesc(&desc);
                // std::cout << "Adapter: "<<desc.Description << std::endl;
                return pAdapter;
            }
            pAdapter->Release();
        }
        pFactory->Release();
        return NULL;
    }

private:
    bool bReady = false;
    bool bQuit = false;
    std::mutex mtx;
    std::thread *pthMsgLoop = NULL;

    IDXGISwapChain *pSwapChain = NULL;
    ID3D11DeviceContext *pContext = NULL;
    ID3D11Texture2D *pBackBuffer = NULL, *pStagingTexture = NULL;
    HANDLE hTimer;
    HANDLE hTimerQueue;
    HANDLE hPresentEvent;


    template <class T>
    using ComPtr = Microsoft::WRL::ComPtr<T>;

    ComPtr<ID3D11DeviceContext> m_pd3dImmediateContext;
    // Direct2D
    ComPtr<ID2D1Factory> m_pd2dFactory;							
    ComPtr<ID2D1RenderTarget> m_pd2dRenderTarget;				
    ComPtr<IDWriteFactory> m_pdwriteFactory;					
    ComPtr<IDXGISurface> surface;

    ComPtr<ID2D1SolidColorBrush> m_pColorBrush;	    
    ComPtr<IDWriteFont> m_pFont;					
    ComPtr<IDWriteTextFormat> m_pTextFormat;		

    ID2D1SolidColorBrush** pColorBrushs;

    // uint8_t *render_surface;
    const int max_size = 1000;
    float* pRect;
    int* pClass;
    int p_num;

    int class_num;
    float* pColors;
    std::vector<std::string> labels = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush" };

    std::vector<std::wstring> wlabels;

    std::string winName="D3D11";

    D2D1_RECT_F rc;
    cv::Mat frame_bgra;
    int size_bgra;
};
