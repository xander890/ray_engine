#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>

using optix::float3;

__device__ float3 approximate_standard_dipole_bssrdf(float r, const ScatteringMaterialProperties& properties)
{
	optix_print("BSSRDF: approximate standard \n");
	float3 A = properties.albedo;
	float3 one_over_l = make_float3(1);
	float3 s = make_float3(3.5f) + 100.0f * pow(A - make_float3(0.33f), make_float3(4.0f));
	float3 exp1 = exp(-one_over_l*s*r);
	float3 exp2 = exp(-one_over_l*s*r / 3.0f);
	float3 R =  A * s * one_over_l * (exp1 + exp2) / (8.0f * M_PIf * M_PIf * r); // Extra pi is to get BSSRDF from reflectance
	return R;
}