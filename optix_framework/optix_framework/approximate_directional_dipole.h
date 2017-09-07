#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>

using optix::float3;

__forceinline__ __device__ float3 approximate_directional_dipole_bssrdf(const float3& _xi, const float3& _ni, const float3& _w12,
	const float3& _xo, const float3& _no, const float3 & _w21,
	const ScatteringMaterialProperties& properties)
{
	optix_print("BSSRDF: approximate directional\n");
	float r = length(_xo - _xi);
	float one_over_r = 1.0f / r;
	float3 A = properties.albedo;
	float3 one_over_l = make_float3(1);
	float3 s = make_float3(3.5f) + 100.0f * pow(A - make_float3(0.33f), make_float3(4.0f));
	float3 exp1 = exp(-one_over_l*s*r);
	float3 exp2 = exp(-one_over_l*s*r / 3.0f);
	float3 R = A * s * one_over_l * (exp1 + exp2) / (8.0f * M_PIf * M_PIf * r); // Extra pi is to get BSSRDF from reflectance

	float3 x = _xo - _xi;
	float3 one_over_dr = s * one_over_l * one_over_r;
	
	float3 additional_terms = make_float3(1.0f) + dot(x, _w12 + _w21) * one_over_dr; //+ dot(x, _w12)*dot(x, _w21) * one_over_dr * one_over_dr;

	return R * additional_terms; // Extra pi is to get BSSRDF from reflectance
}