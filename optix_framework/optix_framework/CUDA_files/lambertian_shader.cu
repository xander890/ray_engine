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


using namespace optix;

// Standard ray variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_cache, prd_cache, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );


rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(uint, ray_traced_reflection, , );

// Material properties (corresponding to OBJ mtl params)
rtTextureSampler<float4, 2> ambient_map;
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;


// Monte carlo variables
rtDeclareVariable(unsigned int, N, , );
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(int, max_splits, , );
rtDeclareVariable(int, use_split, , );
rtDeclareVariable(uint, frame, , );


rtDeclareVariable(float3, eye, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() {
	float3 emission = make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
	 //optix_print("%f %f %f", emission.x,emission.y,emission.z);
	shadow_hit(prd_shadow, emission);
}

__inline__ __device__ float3 sample_procedural_tex(float3 & position_local) 
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


__inline__ __device__ float3 get_k_d()
{
	float3 k_d = make_float3(tex2D(diffuse_map, texcoord.x, texcoord.y));
	//float3 k_d = make_float3(texcoord.x, texcoord.y, 0);
	return k_d;
}

__inline__ __device__ float3 shade_specular(const float3& hit_pos, const float3 & normal, const float3 & light_vector, const float3& light_radiance, const float3 & view)
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

// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void shade()
{
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(normal, -ray.direction, normal);
	float3 k_a = make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
	float3 hit_pos = ray.origin + t_hit * ray.direction;

	hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);
	prd_radiance.flags &= ~(RayFlags::USE_EMISSION); //Unset use emission

	float3 color = make_float3(0.0f);
	color += k_a;  
	//optix_print("%f", k_a.x);
	float3 view = normalize(W);
	uint s = prd_radiance.seed;
	for (int i = 0; i < light_size(); ++i)
	{
		// Diffuse
		
		HitInfo data(hit_pos, normal);
		for (unsigned int i = 0; i < light_size(); i++)
		{
			float3 direct = make_float3(0);
			int M = 20;
			for (int j = 0; j < M; j++)
			{
				float3 light_vector;
				float3 light_radiance;
				int cast_shadows;
				s = lcg(s);
				evaluate_direct_light(data.hit_point, data.hit_normal, light_vector, light_radiance, cast_shadows, s, i);
				float attenuation = 1.0f;
				direct += shade_specular(hit_pos, ffnormal, light_vector, light_radiance, view);
			}
			color += direct / static_cast<float>(M);
		}
	}
	prd_radiance.result = color;
	prd_radiance.seed = s;
}



RT_PROGRAM void shade_path_tracing()
{
	PerRayData_radiance& radiance = (ray.ray_type == dummy_ray_type) ? prd_cache.radiance : prd_radiance;
	optix_print("Lambertian Hit\n");
	float3 k_d = get_k_d();
   float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
   //float3 ffnormal = faceforward(normal, -ray.direction, normal);
   float3 hit_pos = ray.origin + t_hit * ray.direction;

	if (radiance.depth < max_depth)
	{
		uint t = radiance.seed;
		const HitInfo data(hit_pos, normal);
		// Direct illumination
		float3 direct = make_float3(0.0f); 
		for (unsigned int i = 0; i < light_size(); i++)
		{
			for (int j = 0; j < N; j++)
			{
				float3 wi, L; int sh;
				evaluate_direct_light(hit_pos, data.hit_normal, wi, L, sh, t, i);
				direct += L;
			}
		}
		direct /= static_cast<float>(N);

		float3 env = make_float3(0.0f);
		for (int j = 0; j < N; j++)
		{
			float3 wi, L; //int sh;
			//evaluate_environment_light(wi, L, sh, data, t);
			sample_environment(wi, L, data, t);
			float cos_theta = dot(wi, normal);
			if (cos_theta <= 0.0) continue;
			env += L * cos_theta;
		}
		env /= static_cast<float>(N);

		float3 emission = make_float3(0.0f);
		if (radiance.flags & RayFlags::USE_EMISSION)
		{
			// Only the first hit uses emission
			radiance.flags &= ~(RayFlags::USE_EMISSION); //Unset use emission
			emission += make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
			//if (radiance.depth > 0 && emission.x > 0)
			//	optix_print("Emission requested. Path depth %d. Emission %f %f %f", radiance.depth, emission.x, emission.y, emission.z);
		}

		// Indirect illumination
		float prob = dot(k_d, make_float3(0.33333f));
		radiance.flags |= RayFlags::HIT_DIFFUSE_SURFACE;
		float3 indirect = make_float3(0.0f);
		float random = rnd(t);
	    if(random < prob)
		{
			float xi1 = rnd(t);
			float xi2 = rnd(t);
			float3 hemi_vec = sample_hemisphere_cosine(make_float2(xi1, xi2), normal);
			PerRayData_radiance prd;
			prd.depth = radiance.depth + 1;
			prd.flags = radiance.flags;
			prd.seed = t;
			prd.colorband = radiance.colorband;

			optix::Ray ray = optix::make_Ray(hit_pos, hemi_vec, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

			rtTrace(top_object, ray, prd);
			indirect = prd.result / prob * M_PIf; // Cosine cancels out
			radiance.seed = prd.seed;
			radiance.colorband = prd.colorband;

    }
    else
      radiance.seed = t;
	
		optix_print("Lambertian (Bounce: %d) Env: %f %f %f, Dir: %f %f %f, Ind: %f %f %f\n", radiance.depth, env.x, env.y, env.z, direct.x, direct.y, direct.z, indirect.x, indirect.y, indirect.z);
	radiance.result = emission + k_d * M_1_PIf * (env + indirect + direct);
	}
	else
	{
	  radiance.result = make_float3(0.0f);
	}

}