#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>

using optix::float3;

__device__ float3 approximate_standard_dipole_bssrdf(float r, const ScatteringMaterialProperties& properties)
{
	float3 A = properties.approx_property_A;
	float3 s = properties.approx_property_s;
	float3 sr = r * s;
	float c = 0.125*M_1_PIf;
	float3 R =  c / r * A * (exp(-sr) + exp(-sr / 3.0f));
	return R * M_1_PIf;
}