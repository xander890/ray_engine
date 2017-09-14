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
	const float r_sqr = dot(x, x);
	const float r = sqrtf(r_sqr);
	float3 x_norm = x / r;
	const float3 A = properties.approx_property_A;
	const float3 s = properties.approx_property_s;
	const float3 sr = r * s;

	float3 R = c / r * A * (exp(-sr) + exp(-sr / 3.0f));
	float3 S = R * (make_float3(1.0f) + dot(x_norm, w12 + w21) * s + dot(x_norm, w12)*dot(x_norm, w21) * s * s);
	return S * M_1_PIf;
}


__forceinline__ __device__ float3 approximate_directional_dipole_bssrdf(const float3& _xi, const float3& _ni, const float3& _w12,
	const float3& _xo, const float3& _no, const float3 & _w21,
	const ScatteringMaterialProperties& properties)
{
	float3 w21 = -_w21;
	float3 x = _xo - _xi;	
	return approx_bssrdf(x, _w12, w21, properties) / M_PIf; // Extra pi is to get BSSRDF from reflectance
}