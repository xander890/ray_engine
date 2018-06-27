// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011


#include "subsurface_scattering_sampled_common.h"

using namespace optix;

#ifdef ENABLE_NEURAL_NETWORK
_fn float3 get_beam_transmittance(const float depth, const ScatteringMaterialProperties& properties)
{
    return exp(-depth*properties.extinction);
}
#endif

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(unsigned int, samples_per_pixel, , );

rtDeclareVariable(BufPtr<BSSRDFSamplingProperties>, bssrdf_sampling_properties, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.attenuation = 0.0f;
	rtTerminateRay();
}


// Closest hit program for Lambertian shading using the basic light as a directional source
_fn void _shade()
{
	if (prd_radiance.depth > max_depth)
	{
		prd_radiance.result = make_float3(0.0f);
		return;
	}

	TEASampler * sampler = prd_radiance.sampler;

	float3 no = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 xo = ray.origin + t_hit*ray.direction;
	float3 wo = -ray.direction;
	const MaterialDataCommon material = get_material(xo);
	const ScatteringMaterialProperties& props = material.scattering_properties;
	float relative_ior = get_monochromatic_ior(material);
	float recip_ior = 1.0f / relative_ior;
	float reflect_xi = sampler->next1D();
	prd_radiance.result = make_float3(0.0f);

	float3 beam_T = make_float3(1.0f);
	float cos_theta_o = dot(wo, no);
	bool inside = cos_theta_o < 0.0f;
	if (inside)
	{
		beam_T = get_beam_transmittance(t_hit, props);
		float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
		if (sampler->next1D() >= prob) return;
		beam_T /= prob;
		recip_ior = relative_ior;
		cos_theta_o = -cos_theta_o;
        no = -no;
	}

	float3 wt;
	float R;
	refract(wo, no, recip_ior, wt, R);

	R = bssrdf_sampling_properties->show_mode == BSSRDF_SHADERS_SHOW_REFLECTION ? 1.0f : R;
	R = bssrdf_sampling_properties->show_mode == BSSRDF_SHADERS_SHOW_REFRACTION ? 0.0f : R;

    // Trace reflected ray
#if 1
	if (reflect_xi < R)
	{
		float3 wr = -reflect(wo, no);
		PerRayData_radiance prd_reflected = prepare_new_pt_payload(prd_radiance);
		Ray reflected(xo, wr,  RayType::RADIANCE, scene_epsilon);
		rtTrace(top_object, reflected, prd_reflected);
		prd_radiance.result += prd_reflected.result;
	}
#endif
    if (reflect_xi >= R)
	{
		PerRayData_radiance prd_refracted = prepare_new_pt_payload(prd_radiance);

		Ray refracted(xo, wt,  RayType::RADIANCE, scene_epsilon);
		rtTrace(top_object, refracted, prd_refracted);

		prd_radiance.result += prd_refracted.result*beam_T;

		if (!inside)
		{
			float3 L_d = make_float3(0.0f);
			uint N = samples_per_pixel;

			for (uint i = 0; i < N; i++)
			{
				float3 integration_factor = make_float3(1.0f);
				float3 xi, ni;
				bool has_candidate_wi;
				float3 proposed_wi;

				if (!importance_sample_position(xo, no, wo, material, bssrdf_sampling_properties[0], sampler, xi, ni, integration_factor, has_candidate_wi, proposed_wi))
				{
					optix_print("Sample non valid.\n");
					continue;
				}

                if(bssrdf_sampling_properties->exclude_backfaces && dot(no, ni) < 0.0f)
                {
                    continue;
                }

				optix::float3 wi = make_float3(0);
				optix::float3 L_i;

				optix_print("Sampling complete, evaluating light...\n");

#ifdef ENABLE_NEURAL_NETWORK
				optix_assert(has_candidate_wi);
				wi = proposed_wi;
				PerRayData_radiance prd_extra = get_empty();
				prd_extra.flags = RayFlags::USE_EMISSION;
				prd_extra.depth = prd_radiance.depth + 1;
				prd_extra.sampler = sampler;
				optix::Ray ray = optix::make_Ray(xi, wi, RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
				rtTrace(top_object, ray, prd_extra);
				L_i = prd_extra.result;
#else
                sample_light(xi, ni, 0, sampler, wi, L_i); // This returns pre-sampled w_i and L_i
#endif

				// compute direction of the transmitted light

				float3 w12;
				float R12;
				refract(wi, ni, recip_ior, w12, R12);
				float T12 = 1.0f - R12;
				optix_print("Sampling complete, evaluating bssrdf...\n");
				// compute contribution if sample is non-zero
				if (dot(L_i, L_i) > 0.0f)
				{
#ifdef ENABLE_NEURAL_NETWORK
                    float3 S = make_float3(1);
#else
					BSSRDFGeometry geometry;
					geometry.xi = xi;
					geometry.ni = ni;
					geometry.wi = wi;
					geometry.xo = xo;
					geometry.no = no;
					geometry.wo = wo;
					float3 S = bssrdf(geometry, recip_ior, material, BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL, *sampler);
#endif
					L_d += L_i * S * integration_factor;
					optix_print("Sd %e %e %e Ld %f %f %f Li %f %f %f T12 %f int %f\n",  S.x, S.y, S.z, L_d.x, L_d.y, L_d.z, L_i.x, L_i.y, L_i.z, T12, integration_factor);
				}

			}
			prd_radiance.result += L_d / (float)N;
		}
	}
}

RT_PROGRAM void shade() { _shade(); }
RT_PROGRAM void shade_path_tracing() { _shade(); }
