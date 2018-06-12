// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011


#include <device_common_data.h>
#include <light.h>
#include <random.h>
#include <sampling_helpers.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <optical_helper.h>
#include <environment_map.h>

//#define IMPORTANCE_SAMPLE_BRDF
#include <material_device.h>
#include <brdf.h>
#include "device_environment_map.h"


using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );

// Monte carlo variables
rtDeclareVariable(unsigned int, N, , );


// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() {
	float3 emission = make_float3(rtTex2D<float4>(get_material(texcoord).ambient_map, texcoord.x, texcoord.y));
	// optix_print("%f %f %f", emission.x,emission.y,emission.z);
	shadow_hit(prd_shadow, emission);
}


RT_PROGRAM void shade()
{
    const MaterialDataCommon & material = get_material(texcoord);
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 wo = -ray.direction;
    float3 hit_pos = ray.origin + t_hit * ray.direction;

    if (prd_radiance.depth < max_depth)
	{
        float3 k_d = make_float3(rtTex2D<float4>(material.diffuse_map, texcoord.x, texcoord.y));

		float3 emission = make_float3(0.0f);
		if (prd_radiance.flags & RayFlags::USE_EMISSION)
		{
			// Only the first hit uses direct illumination
			prd_radiance.flags &= ~(RayFlags::USE_EMISSION); //Unset use emission
			emission += make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
		}

		// Indirect illumination
		float prob = luminance_NTSC(k_d);
		float3 indirect = make_float3(0.0f);
		float random = prd_radiance.sampler->next1D();

		if (random < prob)
		{
			PerRayData_radiance prd = prd_radiance;
			prd.depth = prd_radiance.depth + 1;
            prd.flags |= RayFlags::USE_EMISSION;

			BRDFGeometry g;
			g.texcoord = texcoord;
			g.wo = wo;
			g.n = normal;

            float3 new_direction, f_d_weighted;
			importance_sample_new_direction_brdf(g, material, *prd_radiance.sampler, new_direction, f_d_weighted);
			optix::Ray ray_t = optix::make_Ray(hit_pos, new_direction, RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
			rtTrace(top_object, ray_t, prd);

			indirect = prd.result * f_d_weighted / fmaxf(1e-6f,prob);
		}

		prd_radiance.result = emission + indirect;
	}
	else
	{
		prd_radiance.result = make_float3(0.0f);
	}
	
}