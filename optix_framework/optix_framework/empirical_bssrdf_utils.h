#pragma once
#include "host_device_common.h"
#include "bssrdf_properties.h"
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

__forceinline__ __device__ void empirical_bssrdf_get_geometry(const BSSRDFGeometry & geometry, float& theta_i, float &r, float& theta_s, float& theta_o, float& phi_o)
{
	float cos_theta_i = dot(geometry.wi, geometry.ni);
	theta_i = acosf(cos_theta_i);

	optix::float3 x = geometry.xo - geometry.xi;
	optix::float3 x_norm = normalize(x);
	float cos_theta_o = dot(geometry.no, geometry.wo);
	optix::float3 x_bar = -normalize(geometry.wi - cos_theta_i * geometry.ni);

	if(fabs(theta_i) <= 1e-6f)
	{
		x_bar = x_norm;
	}

	optix::float3 z_bar = normalize(cross(geometry.ni, x_bar));
	theta_s = -atan2(dot(z_bar, x_norm),dot(x_bar, x_norm));

	// theta_s mirroring.
	if(theta_s < 0) {
		theta_s = abs(theta_s);
		z_bar = -z_bar;
	}

	optix::float3 xo_bar = normalize(geometry.wo - cos_theta_o * geometry.no);
	theta_o = acosf(cos_theta_o);
	phi_o = atan2f(dot(z_bar,xo_bar), dot(x_bar,xo_bar));

	phi_o = normalize_angle(phi_o);
	r = optix::length(x);
			optix_assert(theta_i >= 0 && theta_i <= M_PIf/2);
			optix_assert(theta_s >= 0 && theta_s <= M_PIf);
			optix_assert(theta_o >= 0 && theta_o <= M_PIf/2);
			optix_assert(phi_o >= 0 &&  phi_o < 2*M_PIf);
}