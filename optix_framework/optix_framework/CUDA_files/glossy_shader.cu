// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011


#include <device_common_data.h>
#include <light.h>
#include <random.h>
#include <sampling_helpers.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <structs_device.h>
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
	float3 normal;
	float3 hit_pos;
	normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 wo = -ray.direction;
    float3 k_d = make_float3(rtTex2D<float4>(material.diffuse_map, texcoord.x, texcoord.y));
    float3 brdf_normal = normal;

	hit_pos = ray.origin + t_hit * ray.direction;
	hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);
	float recip_ior = 1.0f / material.relative_ior;

	if (prd_radiance.depth < max_depth)
	{
		HitInfo data(hit_pos, brdf_normal);
		// Direct illumination
		float3 direct = make_float3(0.0f);
		for (int j = 0; j < N; j++)
		{
			float3 wi, L; int sh;
			unsigned int l;
			evaluate_direct_light(hit_pos, brdf_normal, wi, L, sh, prd_radiance.sampler, l);

			BRDFGeometry g;
			g.texcoord = texcoord;
			g.wi = wi;
			g.wo = wo;
			g.n = brdf_normal;

			float3 f_d = brdf(g, recip_ior, material, *prd_radiance.sampler);
			direct += L * f_d;
		}
		direct /= static_cast<float>(N);

		float3 env = make_float3(0.0f);
		for (int j = 0; j < N; j++)
		{
			float3 wi, L; //int sh;
			sample_environment(wi, L, data, prd_radiance.sampler);
			float cos_theta = dot(wi, brdf_normal);
			if (cos_theta <= 0.0) continue;

			BRDFGeometry g;
			g.texcoord = texcoord;
			g.wi = wi;
			g.wo = wo;
			g.n = brdf_normal;

			float3 f_d = brdf(g, recip_ior, material, *prd_radiance.sampler);
			env += L * f_d * cos_theta;
		}
		env /= static_cast<float>(N);
	

		float3 emission = make_float3(0.0f);
		if (prd_radiance.flags & RayFlags::USE_EMISSION)
		{
			// Only the first hit uses direct illumination
			prd_radiance.flags &= ~(RayFlags::USE_EMISSION); //Unset use emission
			emission += make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
		}

		// Indirect illumination
		float prob = luminance_NTSC(k_d);
		prd_radiance.flags |= RayFlags::HIT_DIFFUSE_SURFACE;
		float3 indirect = make_float3(0.0f);
		float random = prd_radiance.sampler->next1D();

		if (random < prob)
		{
			PerRayData_radiance prd = prd_radiance;
			prd.depth = prd_radiance.depth + 1;
			float3 hemi_vec = sample_hemisphere_cosine(prd_radiance.sampler->next2D(), brdf_normal);
			optix::Ray ray_t = optix::make_Ray(hit_pos, hemi_vec,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
			rtTrace(top_object, ray_t, prd);

			BRDFGeometry g;
			g.texcoord = texcoord;
			g.wi = hemi_vec;
			g.wo = wo;
			g.n = brdf_normal;

			float3 f_d = brdf(g, recip_ior, material, *prd_radiance.sampler) * M_PIf;
			indirect = prd.result * f_d / fmaxf(1e-6f,prob); //Cosine cancels out
		}

		prd_radiance.result = emission + direct + env + indirect;
		optix_print("Glossy (Bounce: %d) Env: %f %f %f, Dir: %f %f %f, Ind: %f %f %f\n", prd_radiance.depth, env.x, env.y, env.z, direct.x, direct.y, direct.z, indirect.x, indirect.y, indirect.z);
	}
	else
	{
		prd_radiance.result = make_float3(0.0f);
	}
	
}