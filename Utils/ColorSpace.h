#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <iostream>

typedef enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_Reserved = 3,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470 = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_YCgCo = 8,
    ColorSpaceStandard_BT2020 = 9,
    ColorSpaceStandard_BT2020C = 10
} ColorSpaceStandard;

union BGRA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b, g, r, a;
    } c;
};

union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

union BGRA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t b, g, r, a;
    } c;
};

union RGBA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t r, g, b, a;
    } c;
};



void bgra2bgr(uint8_t *rgba,uint8_t *rgb,int rows,int cols);
void bgr2bgra(uint8_t *rgb,uint8_t *rgba,int rows,int cols);
void bgra2bgrAsync(uint8_t *rgba,uint8_t *rgb,int rows,int cols,cudaStream_t &stream);
void bgr2bgraAsync(uint8_t *rgb,uint8_t *rgba,int rows,int cols,cudaStream_t &stream);
void Bgra2Yuv(uint8_t *bgra,uint8_t *yuv,int rows,int cols);
void Bgra2YuvAsync(uint8_t *bgra,uint8_t *yuv,int rows,int cols,cudaStream_t &stream);
// void Bgra2Yuv1(uint8_t *bgra,uint8_t *yuv,uint8_t *u,int rows,int cols);

