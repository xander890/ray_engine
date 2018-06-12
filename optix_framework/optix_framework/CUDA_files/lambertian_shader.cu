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
#include <camera.h>
#include <material_device.h>
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


rtDeclareVariable(float3, eye, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() {
    const MaterialDataCommon & material = get_material(texcoord);
    float3 emission = make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
	shadow_hit(prd_shadow, emission);
}

_fn float3 sample_procedural_tex(float3 & position_local)
{
	const float3 dims = make_float3(19.0f, 1.3f, 24.7f) * 0.5;
	const float3 black = make_float3(0.1f);
	const float3 white = make_float3(1.0f);

	if (position_local.y < dims.y - 0.01)
		return black;
	if (position_local.x < -dims.x + 1.0f || position_local.x > dims.x - 1.0f || position_local.z < -dims.z + 1.0f || position_local.z > dims.z - 1.0f)
		return black;

	if (position_local.x > -dims.x + 2.0f && position_local.x < dims.x - 2.0f && position_local.z > -dims.z + 2.3f && position_local.z < dims.z - 6.1f)
	{
		position_local = position_local - dims + make_float3(2.0f, 2.3f, 0.0f);
		float3 col = white;
		bool r = (int)(position_local.x / 1.5) % 2 == 0;
		bool c = (int)(position_local.z / 1.5) % 2 == 0;
		if (r && c || !r && !c)
		{
			col = black;
		}

		return col;
	}
	return white;
}


_fn float3 get_k_d()
{
    MaterialDataCommon material = get_material(texcoord);
	float3 k_d = make_float3(rtTex2D<float4>(material.diffuse_map, texcoord.x, texcoord.y));
	return k_d;
}

_fn float3 shade_specular(const float3& hit_pos, const float3 & normal, const float3 & light_vector, const float3& light_radiance, const float3 & view)
{
	const float3 k_d = get_k_d();
	float3 color = light_radiance * k_d * M_1_PIf * max(dot(normal, light_vector), 0.0f);
	// Specular
	//const float3 k_s = make_float3(tex2D(specular_map, texcoord.x, texcoord.y));
	//const float shininess = phong_exp;
	//const float3 half_vector = -normalize(light_vector + view);
	//float ndoth = pow(max(0.0f, dot(normal, half_vector)), shininess);
	//color += light_radiance * k_s * ndoth;
	return color;
}

RT_PROGRAM void shade()
{
	const MaterialDataCommon & material = get_material(texcoord);
	float3 k_d = get_k_d();
	optix_print("Lambertian Hit Kd = %f %f %f\n", k_d.x, k_d.y, k_d.z);
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	//float3 ffnormal = faceforward(normal, -ray.direction, normal);
	float3 hit_pos = ray.origin + t_hit * ray.direction;

	if (prd_radiance.depth < max_depth)
	{
		// Direct illumination
		float3 direct = make_float3(0.0f); 
		for (int j = 0; j < N; j++)
		{
			unsigned int l;
			float3 wi, L; int sh;
			evaluate_direct_light(hit_pos, normal, wi, L, sh, prd_radiance.sampler, l);
			direct += L;
		}

		direct /= static_cast<float>(N);

		float3 env = make_float3(0.0f);
		for (int j = 0; j < N; j++)
		{
			float3 wi, L;
			sample_environment(wi, L, hit_pos, normal, prd_radiance.sampler);
			float cos_theta = dot(wi, normal);
			if (cos_theta <= 0.0) continue;
			env += L * cos_theta;
		}
		env /= static_cast<float>(N);

		float3 emission = make_float3(0.0f);
		if (prd_radiance.flags & RayFlags::USE_EMISSION)
		{
			// Only the first hit uses emission
			prd_radiance.flags &= ~(RayFlags::USE_EMISSION); //Unset use emission
            emission += make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
			//if (radiance.depth > 0 && emission.x > 0)
			//	optix_print("Emission requested. Path depth %d. Emission %f %f %f", radiance.depth, emission.x, emission.y, emission.z);
		}

		// Indirect illumination
		float prob = dot(k_d, make_float3(0.33333f));
		float3 indirect = make_float3(0.0f);
		float random = prd_radiance.sampler->next1D();
	    if(random < prob)
		{
			float3 hemi_vec = sample_hemisphere_cosine(prd_radiance.sampler->next2D(), normal);
			PerRayData_radiance prd = prepare_new_pt_payload(prd_radiance);
			prd.colorband = prd_radiance.colorband;
			
			optix::Ray ray = optix::make_Ray(hit_pos, hemi_vec,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);

			rtTrace(top_object, ray, prd);
			indirect = prd.result / prob * M_PIf; // Cosine cancels out
			prd_radiance.colorband = prd.colorband;
		}
	
		optix_print("Lambertian (Bounce: %d) Env: %f %f %f, Dir: %f %f %f, Ind: %f %f %f\n", prd_radiance.depth, env.x, env.y, env.z, direct.x, direct.y, direct.z, indirect.x, indirect.y, indirect.z);
		prd_radiance.result = emission + k_d * M_1_PIf * (env + indirect + direct);
		optix_print("Res: %f %f %f \n", prd_radiance.result.x, prd_radiance.result.y, prd_radiance.result.z);
	}
	else
	{
	  prd_radiance.result = make_float3(0.0f);
	}

}