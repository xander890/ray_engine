#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>
#include <sampling_helpers.h>

using optix::float3;

// Implementation of bssrdf approximation

__device__ optix::float3 approx_bssrdf(const float3& x, const float3& w12, const float3& w21, const ScatteringMaterialProperties& properties)
{
	const float c = 0.125f*M_1_PIf; // 1/(8 pi)
	float r_sqr = dot(x, x);
	float r = sqrtf(r_sqr);
	float3 r_tr = properties.transport*3.0f*r / properties.extinction;
	float3 S = make_float3(1.0f) + make_float3(dot(x, w12 + w21)) / r_tr + make_float3(dot(x, w12)*dot(x, w21)) / (r_tr*r_tr);
	S *= c*(expf(-r_tr) + expf(-r_tr / 3.0f)) / r_tr;
	return max(S, make_float3(0.0f));
}


__forceinline__ __device__ float3 approximate_directional_dipole_bssrdf(const float3& _xi, const float3& _ni, const float3& _w12,
	const float3& _xo, const float3& _no, const float3 & _w21,
	const ScatteringMaterialProperties& properties)
{
	float3 w21 = -_w21;
	float r = length(_xo - _xi);
	float one_over_r = 1.0f / r;
	float3 A = properties.albedo;
	float3 one_over_l = properties.transport;
	float3 s = burley_scaling_factor_diffuse_mfp_searchlight(A);
	float3 s_over_l = s * one_over_l;

	float3 exp1 = exp(-s_over_l*r);
	float3 exp2 = exp(-s_over_l*r / 3.0f);

	float3 one_over_dr = s_over_l * one_over_r;
	float3 R = A * one_over_dr * (exp1 + exp2) / (8.0f * M_PIf); // Extra pi is to get BSSRDF from reflectance

	float3 x = _xo - _xi;
	
	float3 additional_terms = make_float3(1.0f); 
	additional_terms += max(0.0f, dot(x, _w12 + w21)) * one_over_dr;
	additional_terms += max(0.0f, dot(x, _w12)*dot(x, w21)) * one_over_dr * one_over_dr;
	// do debug buffer
	return R * additional_terms / M_PIf; // Extra pi is to get BSSRDF from reflectance
}