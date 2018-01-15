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

__forceinline__ __device__ void print_v3(const optix::float3 & v)
{
    optix_print("%f %f %f\n", v.x, v.y,v.z);
}

__forceinline__ __device__ bool compare_geometries(const BSSRDFGeometry & g1, const BSSRDFGeometry & g2)
{
	bool e0 = fabsf(optix::length(g1.xi - g2.xi)) < 1e-4;
	bool e1 = fabsf(optix::length(g1.xo - g2.xo)) < 1e-4;
	bool e2 = fabsf(optix::length(g1.ni - g2.ni)) < 1e-4;
	bool e3 = fabsf(optix::length(g1.no - g2.no)) < 1e-4;
	bool e4 = fabsf(optix::length(g1.wi - g2.wi)) < 1e-4;
	bool e5 = fabsf(optix::length(g1.wo - g2.wo)) < 1e-4;
    //optix_print("xi %d, xo %d, ni %d, no %d, wi %d, wo %d\n", e0, e1, e2, e3, e4, e5);
	return e0 & e1 & e2 & e3 & e4 & e5;
}

__forceinline__ __device__ void empirical_bssrdf_build_geometry(const optix::float3& xi, const optix::float3& wi, const optix::float3& n, const float& theta_i, const float &r, const float& theta_s, const float& theta_o, const float& phi_o, BSSRDFGeometry & geometry)
{
    const optix::float3 x = -optix::normalize(wi - dot(wi,n) * n);
	geometry.no = geometry.ni = n;
    optix_assert(fabsf(acosf(dot(wi,n)) - theta_i) < 1e-6);
	geometry.wi = sinf(theta_i) * (-x) + cosf(theta_i) * n;
	const optix::float3 z = cross(n,x);
	const optix::float3 xoxi =  cosf(theta_s) * x +  sinf(theta_s) * (-z);
	geometry.xo = xi + r * xoxi;
	geometry.xi = xi;
	const optix::float3 wo_s = optix::make_float3(sinf(theta_o) * cosf(phi_o), sinf(theta_o) * sinf(phi_o), cosf(theta_o));
    float sign = theta_s < 0? -1 : 1;
	geometry.wo = x * wo_s.x + sign * z * wo_s.y + geometry.no * wo_s.z;
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

    //float theta_s_original = theta_s;
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

//#define TEST_INVERSE
#ifdef TEST_INVERSE
	BSSRDFGeometry gg;
	empirical_bssrdf_build_geometry(geometry.xi, geometry.wi, geometry.ni, theta_i, r, theta_s_original, theta_o, phi_o, gg);
	bool res = compare_geometries(gg, geometry);
	optix_print("Geometries: %s\n", res? "yes" : "no");
#endif
}