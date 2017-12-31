#pragma once
#include "host_device_common.h"

#define USE_OLD_STORAGE

struct EmpiricalParameterBuffer
{
	rtBufferId<float> buffers[5];
};

struct EmpiricalDataBuffer
{
	rtBufferId<float> buffers[3]; //R,G,B
    int test;
};

__forceinline__ __host__ __device__ optix::float2 get_normalized_hemisphere_buffer_coordinates(float theta_o, float phi_o)
{
	const float phi_o_normalized = normalize_angle(phi_o) / (2.0f * M_PIf);
	// Uniform sampling of hemisphere
#ifdef USE_OLD_STORAGE
	const float theta_o_normalized = cosf(theta_o);
#else
	const float theta_o_normalized = 1 - cosf(theta_o);
#endif
	optix_assert(theta_o_normalized >= 0.0f);
	optix_assert(theta_o_normalized < 1.0f);
	optix_assert(phi_o_normalized < 1.0f); 
	optix_assert(phi_o_normalized >= 0.0f);
	return optix::make_float2(phi_o_normalized, theta_o_normalized);
}

__forceinline__ __host__ __device__ optix::float2 get_normalized_hemisphere_buffer_angles(float theta_o_normalized, float phi_o_normalized)
{
	const float phi_o = phi_o_normalized * (2.0f * M_PIf);
	// Uniform sampling of hemisphere
#ifdef USE_OLD_STORAGE
	const float theta_o = acosf(theta_o_normalized);
#else
	const float theta_o = acosf(1 - theta_o_normalized);
#endif
	return optix::make_float2(phi_o, theta_o);
}

