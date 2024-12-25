cmake_minimum_required(VERSION 3.10)
project(yolo_trt_nvcodec)
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME yolotrt)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)



#CUDA
find_package(CUDA REQUIRED)
# set(CUDNN_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")

#OpenCV
SET(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thrid_lib/opencv4.7)
set(OpenCV_FOUND True)

find_package(OpenCV REQUIRED)

#NVTX
set(NVTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NVTX/c)
include_directories(${NVTX_DIR}/include)

#Decode
set(NV_CODEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec)
set(NV_ENC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder)
set(NV_DEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvDecoder)
set(NV_UTIL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Utils)

set(NV_DEC_HDRS
    ${NV_DEC_DIR}/NvDecoder.h
    ${NVCODEC_PUBLIC_INTERFACE_DIR}/cuviddec.h
    ${NVCODEC_PUBLIC_INTERFACE_DIR}/nvcuvid.h
)
set(NV_DEC_SOURCES
    ${NV_DEC_DIR}/NvDecoder.cpp
)
set(NV_ENC_SOURCES
    ${NV_ENC_DIR}/NvEncoder.cpp
    ${NV_ENC_DIR}/NvEncoderCuda.cpp
)

set(NVCODEC_PUBLIC_INTERFACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Interface)

set(NVCODEC_EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/External)

# set(NVCODEC_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Lib/)

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        Set(CUVID_LIB ${CMAKE_CURRENT_SOURCE_DIR}/Lib/x64/nvcuvid.lib)
        set(NVENCODEAPI_LIB ${CMAKE_CURRENT_SOURCE_DIR}/Lib/x64/nvencodeapi.lib)
    else()
        Set(CUVID_LIB ${CMAKE_CURRENT_SOURCE_DIR}/Lib/Win32/nvcuvid.lib)
        set(NVENCODEAPI_LIB ${CMAKE_CURRENT_SOURCE_DIR}/Lib/Win32/nvencodeapi.lib)
    endif()
else()
    find_library(CUVID_LIB nvcuvid)
    find_library(NVENCODEAPI_LIB nvidia-encode)
endif()

#FFMPEG
set(FFMPEG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thrid_lib/ffmpeg-win64-lgpl-shared)
set(GLEW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Lib/glew-2.1.0)
set(GLUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External/freeglut)
set(GLUT_INC ${CMAKE_CURRENT_SOURCE_DIR}/External/freeglut/include)

set(NV_FFMPEG_HDRS ${FFMPEG_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/NvCodec)
include_directories(${NV_DEC_HDRS})
include_directories(${NV_UTIL_DIR})

#Direct2D
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/um)
# include_directories("C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um")

# TensorRT
set(TENSORRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thrid_lib/TensorRT-8.6.1.6)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
    TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

# find_library(CUDNN_LIBRARY libcudnn.so HINTS ${CUDNN_ROOT} lib lib64)
# message("FOUND:${CUDNN_LIBRARY}")

set(LOGCPP ${PROJECT_SOURCE_DIR}/common/logger.cpp)

include_directories(
    ${TENSORRT_INCLUDE_DIR} 
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
)

file(GLOB CUVID_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.cu
)

# message("PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR})

file(COPY ${PROJECT_SOURCE_DIR}/weights DESTINATION ${PROJECT_SOURCE_DIR}/bin)
file(COPY ${PROJECT_SOURCE_DIR}/thrid_lib/TensorRT-8.6.1.6/lib/nvinfer.dll DESTINATION ${PROJECT_SOURCE_DIR}/bin/Release)
file(COPY ${PROJECT_SOURCE_DIR}/thrid_lib/opencv4.7/x64/vc17/bin/opencv_world470.dll DESTINATION ${PROJECT_SOURCE_DIR}/bin/Release)

file(GLOB ffmpeg_dll ${PROJECT_SOURCE_DIR}/thrid_lib/ffmpeg-win64-lgpl-shared/bin/*.dll)
file(COPY ${ffmpeg_dll} DESTINATION ${PROJECT_SOURCE_DIR}/bin/Release)
file(GLOB runtime_dll ${PROJECT_SOURCE_DIR}/thrid_lib/runtime_dll/*.dll)
file(COPY ${runtime_dll} DESTINATION ${PROJECT_SOURCE_DIR}/bin/Release)



cuda_add_executable(${TARGET_NAME} 
    ${LOGCPP}
    src/yolov5_trt.cpp
    src/main.cpp
    src/Decode.cpp
    src/Affine.cu
    src/bbox_calculate.cu
    ${NV_DEC_SOURCES}
    ${NV_ENC_SOURCES}
    ${CUVID_CU}
)

set(AVCODEC_LIB ${FFMPEG_DIR}/lib/avcodec.lib)
set(AVFORMAT_LIB ${FFMPEG_DIR}/lib/avformat.lib)
set(AVUTIL_LIB ${FFMPEG_DIR}/lib/avutil.lib)
set(SWRESAMPLE_LIB ${FFMPEG_DIR}/lib/swresample.lib)
set(FREEGLUT_LIB ${GLUT_DIR}/lib/Release/freeglut.lib)
set(FREEGLUT_STATIC_LIB ${GLUT_DIR}/lib/Release/freeglut_static.lib)
# set(FREEGLUT_STATICD_LIB ${GLUT_DIR}/lib/Debug/freeglut_staticd.lib)
# set(FREEGLUTD_LIB ${GLUT_DIR}/lib/Debug/freeglutd.lib)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(GLEW32_LIB ${GLEW_DIR}/lib/Release/x64/glew32.lib)
    set(GLEW32S_LIB ${GLEW_DIR}/lib/Release/x64/glew32s.lib)
    set(GLEW_DLL_DIR ${GLEW_DIR}/bin/Release/x64/)
    set(GLEW_LIB_DIR ${GLEW_DIR}/lib/Release/x64/)
else()
    set(GLEW32_LIB ${GLEW_DIR}/lib/Release/Win32/glew32.lib)
    set(GLEW32S_LIB ${GLEW_DIR}/lib/Release/Win32/glew32s.lib)
    set(GLEW_DLL_DIR ${GLEW_DIR}/bin/Release/Win32/)
    set(GLEW_LIB_DIR ${GLEW_DIR}/lib/Release/Win32/)
endif()
set(FFMPEG_LIB_DIR ${FFMPEG_DIR}/lib/)
set(FFMPEG_DLL_DIR ${FFMPEG_DIR}/bin/)
set(GLUT_LIB_DIR ${GLUT_DIR}/lib)
set(GLUT_DLL_DIR ${GLUT_DIR}/bin)

set(D3D9_LIBS d3d9.lib dxva2.lib d3dcompiler.lib)
set(D3D11_LIBS d3d11.lib dxgi.lib d3dcompiler.lib d2d1.lib dwrite.lib)

file(GLOB externalLibList 
    ${FFMPEG_LIB_DIR}/*.lib 
    # ${FFMPEG_DLL_DIR}/*.dll
    ${GLUT_LIB_DIR}/Release/*.lib
    # ${GLUT_LIB_DIR}/Debug/*.lib
    # ${GLUT_DLL_DIR}/Release/*.dll
    # ${GLUT_DLL_DIR}/Debug/*.dll
    ${GLEW_LIB_DIR}/*.lib 
    # ${GLEW_DLL_DIR}/*.dll
)

message(${OpenCV_LIBS})



target_include_directories(${TARGET_NAME} PUBLIC
    ${NV_DEC_HDRS}
    ${NVCODEC_PUBLIC_INTERFACE_DIR}
    ${NV_FFMPEG_HDRS}
    ${NVCODEC_EXTERNAL}
    ${GLEW_DIR}/include
    ${GLUT_INC}

)

target_link_libraries(${TARGET_NAME} 
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES} 
    # ${CUDNN_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
    ${CUVID_LIB}
    ${NVENCODEAPI_LIB}
    ${externalLibList}
    ${D3D9_LIBS} 
    ${D3D11_LIBS}
)