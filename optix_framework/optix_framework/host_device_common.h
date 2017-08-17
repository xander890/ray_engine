#pragma once
#define LIGHT_TYPE_DIR 0
#define LIGHT_TYPE_POINT 1
#define LIGHT_TYPE_SKY 2
#define LIGHT_TYPE_AREA 3
#define LIGHT_TYPE_COUNT 4

#define RAY_TYPE_RADIANCE 0
#define RAY_TYPE_SHADOW 1
#define RAY_TYPE_DUMMY 2
#define RAY_TYPE_COUNT 3

template<typename T, int Dim=1>
struct BufPtr
{
    rtBufferId<T, Dim> buf;
    __device__ T* operator->() { return &buf[0]; }
#ifndef __CUDACC__
    __host__ explicit BufPtr(rtBufferId<T, Dim> id) : buf(id) {}
#endif
};