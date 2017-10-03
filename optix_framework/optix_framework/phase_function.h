#pragma once
#include "host_device_common.h"
#include "math_helpers.h"
#include "random.h"

__inline__ __device__ optix::float3 sample_HG(optix::float3& forward, float g, optix::uint& t)
{
	float xi = rnd(t);
	float cos_theta;
	if (fabs(g) < 1.0e-3f)
		cos_theta = 1.0f - 2.0f*xi;
	else
	{
		float two_g = 2.0f*g;
		float g_sqr = g*g;
		float tmp = (1.0f - g_sqr) / (1.0f - g + two_g*xi);
		cos_theta = 1.0f / two_g*(1.0f + g_sqr - tmp*tmp);
	}
	float phi = 2.0f*M_PIf*rnd(t);

	// Calculate new direction as if the z-axis were the forward direction
	float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
	optix::float3 v = optix::make_float3(sin_theta*cosf(phi), sin_theta*sinf(phi), cos_theta);

	// Rotate from z-axis to actual normal and return
	rotate_to_normal(forward, v);
	return v;
}

__inline__ __device__ float eval_HG(float cos_alpha, float g)
{
	float den = rsqrtf(1 + g * g - 2 * g * cos_alpha);
	return 0.25f * M_1_PIf * den * den * den;
}