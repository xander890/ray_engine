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
	float3 x = _xo - _xi;	
	optix_print("w21 dir, MUST BE NEGATIVE %f\n", dot(w21, _no));
	return approx_bssrdf(x, _w12, w21, properties) / M_PIf; // Extra pi is to get BSSRDF from reflectance
}