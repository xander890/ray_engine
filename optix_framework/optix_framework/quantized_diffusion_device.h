#pragma once
#include "device_common.h"
#include <scattering_properties.h>
#include <optix_device_utils.h>
#include "phase_function_device.h"
#include "quantized_diffusion_helpers.h"

using optix::float3;

rtDeclareVariable(BufPtr1D<QuantizedDiffusionProperties>, qd_properties, , );

_fn float single_approx(optix::float3 xi, optix::float3 ni, optix::float3 w12, optix::float3 xo, optix::float3 no, optix::float4 props) {
  float sigma_s = props.x; float sigma_a = props.y; float g = props.z;
  float sigma_t = sigma_s + sigma_a;
  float sigma_s_p = sigma_s*(1.0f - g);
  float sigma_t_p = sigma_s_p + sigma_a;
  float mu0 = abs(dot(no, w12));
  float d1 = mu0/(3.0f*sigma_t_p);// 1.0/sigma_t_p;\n //
  optix::float3 xs = xi + w12*d1;
  optix::float3 w21 = xo - xs;
  float d2 = length(w21);
  w21 /= d2;
  return sigma_s_p*d1*phase_HG(dot(w12, w21), g)*exp(-sigma_t_p*d1 - sigma_t*d2)/(d2*d2);
}

_fn float3 quantized_diffusion_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior, const MaterialDataCommon& material, unsigned int flags, TEASampler & sampler)
{
    const ScatteringMaterialProperties& properties = material.scattering_properties;
	optix::float4 C = optix::make_float4(properties.C_phi_norm, properties.C_phi, properties.C_E, properties.A);
	float dist = optix::length(geometry.xo - geometry.xi);

	optix::float3 w12, w21;
	float R12, R21;
	bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;

	refract(geometry.wi, geometry.ni, recip_ior, w12, R12);
	refract(geometry.wo, geometry.no, recip_ior, w21, R21);
	optix::float3 res;
	float F = include_fresnel_out? (1 - R21) : 1.0f;

	optix_print("QD : %d %f %d\n", qd_properties->use_precomputed_qd, qd_properties->max_dist_bssrdf, qd_properties->precomputed_bssrdf_size);
	if (qd_properties->use_precomputed_qd == 1)
	{
		const int idx = int(optix::clamp(dist / qd_properties->max_dist_bssrdf, 0.0f, 1.0f) * qd_properties->precomputed_bssrdf_size);
		res = qd_properties->precomputed_bssrdf[idx];

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
		optix::get_channel(k, res) += single_approx(geometry.xi, geometry.ni, w12, geometry.xo, geometry.no, props);
	}
	return res * (1 - R12) * F;

}
