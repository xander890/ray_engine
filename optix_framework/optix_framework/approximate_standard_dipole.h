#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>

using optix::float3;

__device__ float3 approximate_standard_dipole_bssrdf(float r, const ScatteringMaterialProperties& properties)
{
	float3 A = properties.albedo;
	float3 one_over_l = properties.transport;
	float3 s = burley_scaling_factor_diffuse_mfp_searchlight(A);
	float3 s_over_l = s * one_over_l;
	float3 exp1 = exp(-s_over_l*r);
	float3 exp2 = exp(-s_over_l*r / 3.0f);
	float3 R =  A * s_over_l * (exp1 + exp2) / (8.0f * M_PIf * r);
	return R / M_PIf; // BSSRDF from reflectance
}