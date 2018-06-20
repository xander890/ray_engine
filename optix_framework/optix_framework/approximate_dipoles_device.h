#pragma once
#include <device_common.h>
#include <scattering_properties.h>
#include <math_utils.h>
#include <bssrdf_common.h>
#include "material_common.h"
using optix::float3;

rtDeclareVariable(ApproximateBSSRDFProperties, approx_std_bssrdf_props, , );

__device__ float3 approximate_standard_dipole_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material, unsigned int flags, TEASampler & sampler)
{
	bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;

    float r = optix::length(geometry.xo - geometry.xi);
	float3 A = approx_std_bssrdf_props.approx_property_A;
	float3 s = approx_std_bssrdf_props.approx_property_s;
	float3 sr = r * s;
	float c = 0.125*M_1_PIf;
	float3 R =  A * c * s / r * (exp(-sr) + exp(-sr / 3.0f));

	float3 w12, w21;
	float R12, R21;
	refract(geometry.wi, geometry.ni, recip_ior, w12, R12);
	refract(geometry.wo, geometry.no, recip_ior, w21, R21);

	float F = include_fresnel_out? (1 - R21) : 1.0f;
	return R * M_1_PIf * (1 - R12) * F;
}

// Implementation of bssrdf approximation
_fn float3 approximate_directional_dipole_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
												 const MaterialDataCommon& material, unsigned int flags, TEASampler & sampler)
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
	float3 R = approximate_standard_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
	float3 S = R * (optix::make_float3(1.0f) + dot(x_norm, w12 + w21) * s + dot(x_norm, w12)*dot(x_norm, w21) * s * s);
	return S; // R is already divided by the extra PI
}
