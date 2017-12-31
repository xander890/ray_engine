#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <bssrdf_properties.h>
#include "material.h"

using optix::float3;

__device__ float3 standard_dipole_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material, unsigned int flags = BSSRDFFlags::NO_FLAGS, TEASampler * sampler = nullptr)
{
    const ScatteringMaterialProperties& properties = material.scattering_properties;
    float dist = optix::length(geometry.xo - geometry.xi);
	optix_print("BSSRDF: standard\n");
	float3 real_source = properties.three_D*properties.three_D;
	float3 extrapolation = 4.0f*properties.A*properties.D;
	float3 virtual_source = extrapolation*extrapolation;
	float3 corrected_mean_free = properties.three_D + extrapolation;
	float3 r_sqr = make_float3(dist*dist);

	// Compute distances to dipole sources
	float3 d_r_sqr = r_sqr + real_source;
	float3 d_r = make_float3(sqrtf(d_r_sqr.x), sqrtf(d_r_sqr.y), sqrtf(d_r_sqr.z));
	float3 d_v_sqr = r_sqr + virtual_source;
	float3 d_v = make_float3(sqrtf(d_v_sqr.x), sqrtf(d_v_sqr.y), sqrtf(d_v_sqr.z));

	// Compute intensities for dipole sources
	float3 tr_r = properties.transport*d_r;
	float3 S_r = properties.three_D*(1.0f + tr_r) / (d_r_sqr*d_r);
	S_r *= expf(-tr_r);
	float3 tr_v = properties.transport*d_v;
	float3 S_v = corrected_mean_free*(1.0f + tr_v) / (d_v_sqr*d_v);
	S_v *= expf(-tr_v);

	optix::float3 _w12, w21;
	float R12, R21;
	bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;

	refract(geometry.wi, geometry.ni, recip_ior, _w12, R12);
	refract(geometry.wo, geometry.no, recip_ior, w21, R21);
	float F = include_fresnel_out? (1 - R21) : 1.0f;

	return (S_r + S_v) * properties.reducedAlbedo / (4.0f * M_PIf * M_PIf) * (1 - R12) * F;
}
