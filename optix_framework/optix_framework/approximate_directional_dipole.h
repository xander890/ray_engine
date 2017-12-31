#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <math_helpers.h>
#include <sampling_helpers.h>
#include <approximate_standard_dipole.h>

using optix::float3;

// Implementation of bssrdf approximation
__forceinline__ __device__ float3 approximate_directional_dipole_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material, unsigned int flags = BSSRDFFlags::NO_FLAGS, TEASampler * sampler = nullptr)
{
    float3 w12, w21; 
	float R12, R21;
	refract(geometry.wi, geometry.ni, recip_ior, w12, R12);
	refract(geometry.wo, geometry.no, recip_ior, w21, R21);
    w21 = -w21;

	float3 x = geometry.xo - geometry.xi;	
	const float r_sqr = dot(x, x);
	const float r = sqrtf(r_sqr);
	float3 x_norm = x / r;
	const float3 s = approx_std_bssrdf_props.approx_property_s;
	float3 R = approximate_standard_dipole_bssrdf(geometry, recip_ior, material);
	float3 S = R * (make_float3(1.0f) + dot(x_norm, w12 + w21) * s + dot(x_norm, w12)*dot(x_norm, w21) * s * s);
	return S; // R is already divided by the extra PI
}
