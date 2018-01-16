// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#define TRANSMIT
#define REFLECT 

#include <device_common_data.h>
#include <math_helpers.h>
#include <random.h>
#include <bssrdf.h>

#include <optical_helper.h>
#include <structs.h>
#include <ray_trace_helpers.h>
#include <scattering_properties.h>
#include <material_device.h>

using namespace optix;

//#define REFLECT

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// SS properties

// Variables for shading
rtDeclareVariable(BufPtr<PositionSample>, sampling_output_buffer, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

//rtDeclareVariable(unsigned int, bssrdf_enabled, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = 0.0f;
  rtTerminateRay();
}  

// Closest hit program for Lambertian shading using the basic light as a directional source
RT_PROGRAM void shade() 
{
    if (prd_radiance.depth > max_depth)
    {
        prd_radiance.result = make_float3(0.0f);
        return;
    }

    float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 xo = ray.origin + t_hit*ray.direction;
    float3 wo = -ray.direction;
    float3 no = faceforward(n, wo, n);
    const MaterialDataCommon & material = get_material(xo);
    const ScatteringMaterialProperties& props = material.scattering_properties;
    float recip_ior = 1.0f / material.relative_ior;
    float reflect_xi = prd_radiance.sampler->next1D();
    prd_radiance.result = make_float3(0.0f);

#ifdef TRANSMIT
    float3 beam_T = make_float3(1.0f);
    float cos_theta_o = dot(wo, n);
    bool inside = cos_theta_o < 0.0f;
    if (inside)
    {
		beam_T = get_beam_transmittance(t_hit, props);
        float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
        if (prd_radiance.sampler->next1D() >= prob) return;
        beam_T /= prob;
        recip_ior = material.relative_ior;
        cos_theta_o = -cos_theta_o;
    }

	float3 wt;
	float R;
	refract(wo, n, recip_ior, wt, R);

    if (reflect_xi >= R)
    {
		PerRayData_radiance prd_refracted = prepare_new_pt_payload(prd_radiance);
		 
		Ray refracted(xo, wt,  RayType::RADIANCE, scene_epsilon);
		rtTrace(top_object, refracted, prd_refracted);
		prd_radiance.result += prd_refracted.result*beam_T;

        if (!inside)
        {
#else
    float cos_theta_o = dot(wo, no);
    float R = fresnel_R(cos_theta_o, recip_ior);
#endif

    float chosen_transport_rr = get_sampling_mfp(props);
    float3 accumulate = make_float3(0.0f);
    uint N = sampling_output_buffer.size();

    for (uint i = 0; i < N; ++i)
    {
        PositionSample& sample = sampling_output_buffer[i];

        // compute direction of the transmitted light
        const float3& wi = sample.dir;

#ifdef TEST_SAMPLING
		accumulate += make_float3(TEST_SAMPLING_W)*sample.L;
#else
        // compute contribution if sample is non-zero
        if (dot(sample.L, sample.L) > 0.0f)
        {
            // Russian roulette
            float dist = length(xo - sample.pos);
            float exp_term = exp(-dist * chosen_transport_rr);
            if (prd_radiance.sampler->next1D() < exp_term)
            {
                 BSSRDFGeometry geometry;
                 geometry.xi = sample.pos;
                 geometry.ni = sample.normal;
                 geometry.wi = wi;
                 geometry.xo = xo;
                 geometry.no = no;
                 geometry.wo = wo;
                 accumulate += sample.L*bssrdf(geometry, recip_ior, material, BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL, *prd_radiance.sampler) / exp_term;
            }
        }
#endif
    }
#ifdef TRANSMIT
    prd_radiance.result += accumulate / (float)N;
        }
    }
#else
    float T21 = 1.0f - R;
    prd_radiance.result += T21*accumulate / (float)N;
#endif
#ifdef REFLECT
    // Trace reflected ray
    if (reflect_xi < R)
    {
		float3 wr = -reflect(wo, no);
        PerRayData_radiance prd_reflected = prepare_new_pt_payload(prd_radiance);
		Ray reflected(xo, wr,  RayType::RADIANCE, scene_epsilon);
        rtTrace(top_object, reflected, prd_reflected);	
        prd_radiance.result += prd_reflected.result;
    }
#endif
}
