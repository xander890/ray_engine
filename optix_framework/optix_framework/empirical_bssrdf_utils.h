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

__forceinline__ __device__ bool compare_geometries(const BSSRDFGeometry & g1, const BSSRDFGeometry & g2) {
    bool e0 = fabsf(optix::length(g1.xi - g2.xi)) < 1e-4;
    bool e1 = fabsf(optix::length(g1.xo - g2.xo)) < 1e-4;
    bool e2 = fabsf(optix::length(g1.ni - g2.ni)) < 1e-4;
    bool e3 = fabsf(optix::length(g1.no - g2.no)) < 1e-4;
    bool e4 = fabsf(optix::length(g1.wi - g2.wi)) < 1e-4;
    bool e5 = fabsf(optix::length(g1.wo - g2.wo)) < 1e-4;
    optix_print("xi %d, xo %d, ni %d, no %d, wi %d, wo %d\n", e0, e1, e2, e3, e4, e5);
    return e0 & e1 & e2 & e3 & e4 & e5;
}

__forceinline__ __device__ optix::float3 rodrigues(const optix::float3 & v,const optix::float3 & axis, const float angle)
{
    return v * cos(angle) + cross(axis, v) * sin(angle) + axis * dot(axis, v) * (1 - cos(angle));
}

__forceinline__ __device__ void empirical_bssrdf_build_geometry_from_exit(const optix::float3& xo, const optix::float3& wo, const optix::float3& no, const float& theta_i, const float &r, const float& theta_s, const float& theta_o, const float& phi_o, BSSRDFGeometry & geometry)
{
    geometry.xo = xo;
    geometry.no = geometry.ni = no;
    geometry.wo = wo;

    optix::float3 projected = normalize(wo - dot(no, wo) * no);
    optix::float3 tangent, bitangent;

    if(optix::length(projected) < 1e-6)
	{
		create_onb(no,tangent,bitangent); // Everything should be symmetric, so we do not care of the choice of orthonormal basis.
	}
	else
	{
        float sign = theta_s > 0? -1 : 1;
        tangent = normalize(projected * cos(phi_o) - sign * cross(no, projected) * sin(phi_o));
        optix_assert(abs(dot(tangent, no)) < 1e-4);
        optix_assert(abs(dot(projected, no)) < 1e-4);
        bitangent = cross(no, tangent);
	}

    optix::float3 x_vec = cosf(theta_s) * tangent + sinf(theta_s) * bitangent;
    geometry.xi = geometry.xo - r * x_vec;
    geometry.wi = sinf(theta_i) * (-tangent) + cosf(theta_i) * no;
}


__forceinline__ __device__ void empirical_bssrdf_build_geometry(const optix::float3& xi, const optix::float3& wi, const optix::float3& n, const float& theta_i, const float &r, const float& theta_s, const float& theta_o, const float& phi_o, BSSRDFGeometry & geometry)
{
    optix::float3 tangent = -(wi - dot(wi,n) * n);
    optix::float3 bitangent;
    if(optix::length(tangent) < 1e-6)
    {
        create_onb(n,tangent,bitangent); // Everything should be symmetric, so we do not care of the choice of orthonormal basis.
    }
    else
    {
        tangent = optix::normalize(tangent);
        bitangent = cross(n,tangent);
    }

    geometry.xi = xi;
	geometry.no = geometry.ni = n;
    geometry.wi = wi;
    optix_assert(fabsf(acosf(dot(wi,n)) - theta_i) < 1e-6);

	const optix::float3 xoxi =  cosf(theta_s) * tangent +  sinf(theta_s) * bitangent;
	geometry.xo = xi + r * xoxi;

	const optix::float3 wo_s = optix::make_float3(sinf(theta_o) * cosf(phi_o), sinf(theta_o) * sinf(phi_o), cosf(theta_o));
    float sign = theta_s > 0? -1 : 1;
	geometry.wo = tangent * wo_s.x + sign * bitangent * wo_s.y + geometry.no * wo_s.z;
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
	theta_s = atan2(dot(z_bar, x_norm),dot(x_bar, x_norm));

    float theta_s_original = theta_s;
	// theta_s mirroring.
    const optix::float3 x_h = cross(z_bar,geometry.no);
    optix::float3 z_h = cross(geometry.no,x_h);

	// The standard bssrdf is defined with theta_s < 0 in mind. In this case, we mirror.
	// See fig. 4-d of the original paper for reference.
	if(theta_s > 0) {
        z_h = -z_h;
	}
	theta_s = abs(theta_s);

	optix::float3 xo_bar = normalize(geometry.wo - cos_theta_o * geometry.no);
	theta_o = acosf(cos_theta_o);
	phi_o = atan2f(dot(z_h,xo_bar), dot(x_h,xo_bar));

	phi_o = normalize_angle(phi_o);
	r = optix::length(x);

    optix_assert(theta_i >= 0 && theta_i <= M_PIf/2);
	optix_assert(theta_s >= 0 && theta_s <= M_PIf);
	optix_assert(theta_o >= 0 && theta_o <= M_PIf/2);
	optix_assert(phi_o >= 0 &&  phi_o < 2*M_PIf);

#define TEST_INVERSE
#ifdef TEST_INVERSE
	BSSRDFGeometry gg;
	empirical_bssrdf_build_geometry(geometry.xi, geometry.wi, geometry.ni, theta_i, r, theta_s_original, theta_o, phi_o, gg);
	bool res = compare_geometries(gg, geometry);
	optix_print("Geometries in: %s\n", res? "yes" : "no");
    BSSRDFGeometry gg2;
    empirical_bssrdf_build_geometry_from_exit(geometry.xo, geometry.wo, geometry.no, theta_i, r, theta_s_original, theta_o, phi_o, gg2);
    bool res2 = compare_geometries(gg2, geometry);
    optix_print("Geometries out: %s\n", res2? "yes" : "no");
#endif
}