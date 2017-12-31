#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>

using optix::float3;

rtDeclareVariable(ApproximateBSSRDFProperties, approx_std_bssrdf_props, , );

__device__ float3 approximate_standard_dipole_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material)
{
    float r = optix::length(geometry.xo - geometry.xi);
	float3 A = approx_std_bssrdf_props.approx_property_A;
	float3 s = approx_std_bssrdf_props.approx_property_s;
	float3 sr = r * s;
	float c = 0.125*M_1_PIf;
	float3 R =  A * c * s / r * (exp(-sr) + exp(-sr / 3.0f));
	return R * M_1_PIf;
}
