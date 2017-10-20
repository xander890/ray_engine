#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <optix_helpers.h>
#include "phase_function.h"
#include "quantized_diffusion_helpers.h"

using optix::float3;

__device__ __forceinline__ float single_approx(optix::float3 xi, optix::float3 ni, optix::float3 w12, optix::float3 xo, optix::float3 no, optix::float4 props) {
  float sigma_s = props.x; float sigma_a = props.y; float g = props.z; float eta = props.w;
  float sigma_t = sigma_s + sigma_a;
  float sigma_s_p = sigma_s*(1.0 - g);
  float sigma_t_p = sigma_s_p + sigma_a;
  float mu0 = abs(dot(no, w12));
  float d1 = mu0/(3.0*sigma_t_p);// 1.0/sigma_t_p;\n //
  optix::float3 xs = xi + w12*d1;
  optix::float3 w21 = xo - xs;
  float d2 = length(w21);
  w21 /= d2;
  return sigma_s_p*d1*phase_HG(dot(w12, w21), g)*exp(-sigma_t_p*d1 - sigma_t*d2)/(d2*d2);
}

#define USE_PRECOMPUTED

__device__ __forceinline__ float3 quantized_diffusion_bssrdf(const float3& xi, const float3& ni, const float3& w12,
	const float3& xo, const float3& no,
	const ScatteringMaterialProperties& properties)
{
	optix::float4 C = make_float4(properties.C_phi_norm, properties.C_phi, properties.C_E, properties.A);
	float dist = optix::length(xo - xi);
	
	optix::float3 res;

	optix_print("QD : %d %f %d\n", properties.use_precomputed_qd, properties.max_dist_bssrdf, properties.precomputed_bssrdf_size);
	if (properties.use_precomputed_qd == 1)
	{
		const int idx = int(clamp(dist / properties.max_dist_bssrdf, 0.0f, 1.0f) * properties.precomputed_bssrdf_size);
		res = properties.precomputed_bssrdf[idx];

	}
	else
	{
		for (int k = 0; k < 3; k++)
		{
			optix::float4 props = optix::make_float4(optix::get_channel(k, properties.scattering), optix::get_channel(k, properties.absorption), optix::get_channel(k, properties.meancosine), 1.0f);
			optix::get_channel(k, res) = quantized_diffusion<float>(dist, props, C);
		}
	}

	for (int k = 0; k < 3; k++)
	{
		optix::float4 props = optix::make_float4(optix::get_channel(k, properties.scattering), optix::get_channel(k, properties.absorption), optix::get_channel(k, properties.meancosine), 1.0f);
		optix::get_channel(k, res) += single_approx(xi, ni, w12, xo, no, props);
	}
	return res;

}