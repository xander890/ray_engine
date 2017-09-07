#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>

using optix::float3;

__device__ float3 standard_dipole_bssrdf(float dist, const ScatteringMaterialProperties& properties)
{
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
	return (S_r + S_v) * properties.reducedAlbedo / (4.0f * M_PIf * M_PIf);
}