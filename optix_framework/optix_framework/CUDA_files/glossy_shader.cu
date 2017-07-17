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
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
#include <merl_common.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, "");
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float, ior, , );
rtDeclareVariable(uint, ray_traced_reflection, , );

// Material properties (corresponding to OBJ mtl params)
rtTextureSampler<float4, 2> ambient_map;
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;

// Monte carlo variables
rtDeclareVariable(unsigned int, N, , );

rtDeclareVariable(int, window_width, , );
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(int, max_splits, , );
rtDeclareVariable(int, use_split, , );
rtDeclareVariable(uint, frame, , );


rtDeclareVariable(float, exponent_blinn, , );
rtDeclareVariable(optix::float2, exponent_aniso, , );
rtDeclareVariable(optix::float3, object_x_axis, , );
rtDeclareVariable(optix::float3, merl_brdf_multiplier, , );

rtBuffer<float, 1> merl_brdf_buffer;
rtDeclareVariable(uint, has_merl_brdf, , ) = 0;
rtDeclareVariable(float3, eye, , );

// Glossy BRDF functions


__device__ __inline__
float geometric_term_torrance_sparrow(const optix::float3& n, const optix::float3& wi, const optix::float3& wo, const optix::float3& wh)
{
	float n_dot_h = fabsf(dot(n, wh));
	float n_dot_o = fabsf(dot(n, wo));
	float n_dot_i = fabsf(dot(n, wi));
	float i_dot_o = fabsf(dot(wo, wh));
	float min_io = fminf(n_dot_o, n_dot_i);
	return fminf(1.0f, min_io * n_dot_h * 2.0f / i_dot_o);
}


__device__ __inline__
float blinn_microfacet_distribution(const optix::float3& n, const optix::float3& brdf_normal)
{
	float cos_theta = fabsf(dot(n, brdf_normal));
	float D = 0.5f * M_1_PIf * (phong_exp + 2.0f) * pow(cos_theta, phong_exp);
	return D;
}

__device__ __inline__ float torrance_sparrow_brdf(const optix::float3 & n, const optix::float3 & wi, const optix::float3 & wo, float ior)
{
	float cos_o = dot(n, wo);
	float cos_i = dot(n, wi);
	optix::float3 brdf_normal = normalize(wi + wo);
	float cos_brdf_normali = dot(wi, brdf_normal);
	float cos_brdf_normalo = dot(wo, brdf_normal);
	if (cos_brdf_normalo / cos_o <= 0.0f || cos_brdf_normali / cos_i <= 0.0f)
		return 0.0f;

	float D = blinn_microfacet_distribution(n, brdf_normal);
	float G = geometric_term_torrance_sparrow(n, wi, wo, brdf_normal);
	float F = fresnel_R(cos_o, ior);
	float S = 4.0f * cos_o * cos_i;
	return abs(D * F * G / S);
}

__device__ __inline__ float torrance_sparrow_brdf_sampled(const optix::float3 & n, const optix::float3 & wi, const optix::float3 & wo, const optix::float3 & brdf_normalj, float ior)
{

	float cos_o = dot(n, wo);
	float cos_i = dot(n, wo);
	float cos_brdf_normali = dot(wi, brdf_normalj);
	float cos_h = dot(n, brdf_normalj);
	if (cos_o <= 0.0f || cos_h <= 0.0f)
		return 0.0f;
	float F = fresnel_R(cos_brdf_normali, ior);
	return min(2.0f * cos_i / cos_o, min(cos_h / (cos_o * cos_h), 2.0f)) * F;
}

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() {
	//float3 emission = make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
	// optix_print("%f %f %f", emission.x,emission.y,emission.z);
	//shadow_hit(prd_shadow, emission);
}

__inline__ __device__ float3 get_importance_sampled_brdf(const float3& hit_pos, float3 & normal, const float3 & wi, const float3 & new_normal, const float3 & out_v)
{
	float3 k_d = make_float3(tex2D(diffuse_map, texcoord.x, texcoord.y));
	float3 k_s = make_float3(tex2D(specular_map, texcoord.x, texcoord.y));
	float3 f_d = k_d * M_1_PIf;
	return f_d + torrance_sparrow_brdf_sampled(normal, normalize(wi), normalize(out_v), new_normal, ior) * k_s;
}

__inline__ __device__ float3 get_brdf(const float3& hit_pos, float3 & normal, const float3 & wi, const float3 & out_v)
{
	float3 f;
	if (has_merl_brdf == 1)
	{
		f = merl_brdf_multiplier * lookup_brdf_val(merl_brdf_buffer, normal, normalize(wi), normalize(out_v));
	}
	else
	{
		float3 k_d = make_float3(tex2D(diffuse_map, texcoord.x, texcoord.y));
		float3 k_s = make_float3(tex2D(specular_map, texcoord.x, texcoord.y));
		float3 f_d = k_d * M_1_PIf;
		f = f_d + torrance_sparrow_brdf(normal, normalize(wi), normalize(out_v), ior) * k_s;
	}
	return f;
}


// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void shade()
{
	//float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	//float3 ffnormal = faceforward(normal, -ray.direction, normal);
	//float3 k_a = make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
	//float3 k_s = make_float3(tex2D(specular_map, texcoord.x, texcoord.y));
	//float3 hit_pos = ray.origin + t_hit * ray.direction;
	//hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);

	//float3 color = make_float3(0.0f);
	////color += ambient_light_color * k_a;  
	//float3 view = -normalize(W);
	//for (unsigned int i = 0; i < light_size(); ++i)
	//{
	//	// Diffuse
	//	float3 light_vector;
	//	float3 light_radiance;
	//	int cast_shadows;

	//	HitInfo data(hit_pos, normal);
	//	unsigned int s = 0;
	//	evaluate_direct_light(data.hit_point, data.hit_normal, light_vector, light_radiance, cast_shadows, s, i);

	//	float attenuation = 1.0f;
	//	if (cast_shadows)
	//	{
	//		attenuation = trace_shadow_ray(hit_pos, -light_vector, scene_epsilon, RT_DEFAULT_MAX);
	//	}

	//	if (attenuation > 0.0f)
	//	{
	//		float f = torrance_sparrow_brdf(ffnormal, -light_vector, view, ior);

	//		color += k_s * f;
	//	}
	//}
	//prd_radiance.result = color;
}

RT_PROGRAM void shade_path_tracing()
{

	float3 normal;
	float3 hit_pos;
	normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 wo = -ray.direction;
	float3 k_d = make_float3(tex2D(diffuse_map, texcoord.x, texcoord.y));
	float3 brdf_normal = normal;

	hit_pos = ray.origin + t_hit * ray.direction;
	hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);

	if (prd_radiance.depth < max_depth)
	{
		uint t = prd_radiance.seed;
		HitInfo data(hit_pos, brdf_normal);

		// Direct illumination
		float3 direct = make_float3(0.0f);
		for (unsigned int i = 0; i < light_size(); i++)
		{
			for (int j = 0; j < N; j++)
			{
				float3 wi, L; int sh;
				evaluate_direct_light(data.hit_point, data.hit_normal, wi, L, sh,t, i);
				
				float3 f_d = get_brdf(hit_pos, brdf_normal, wi, wo);
				direct += L * f_d;
			}
		}
		direct /= static_cast<float>(N);

		float3 env = make_float3(0.0f);
		for (int j = 0; j < N; j++)
		{
			float3 wi, L; //int sh;
			//evaluate_environment_light(wi, L, sh, data, t);
			sample_environment(wi, L, data, t);
			float cos_theta = dot(wi, brdf_normal);
			if (cos_theta <= 0.0) continue;
			float3 f_d = get_brdf(hit_pos, brdf_normal, wi, wo);
			env += L * f_d * cos_theta;
		}
		env /= static_cast<float>(N);
	

		float3 emission = make_float3(0.0f);
		if (prd_radiance.flags & RayFlags::USE_EMISSION)
		{
			// Only the first hit uses direct illumination
			prd_radiance.flags &= ~(RayFlags::USE_EMISSION); //Unset use emission
			emission += make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
		}

		// Indirect illumination
		float prob = luminance_NTSC(k_d);
		prd_radiance.flags |= RayFlags::HIT_DIFFUSE_SURFACE;
		float3 indirect = make_float3(0.0f);
		float random = rnd(t);

		if (random < prob)
		{
			PerRayData_radiance prd = prd_radiance;
			prd.depth = prd_radiance.depth + 1;
			prd.flags = prd_radiance.flags;
			prd.seed = t;
			float xi1 = rnd(t);
			float xi2 = rnd(t);
			float3 hemi_vec = sample_hemisphere_cosine(make_float2(xi1, xi2), brdf_normal);
			optix::Ray ray_t = optix::make_Ray(hit_pos, hemi_vec, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
			rtTrace(top_object, ray_t, prd);
			float3 f_d = get_brdf(hit_pos, brdf_normal, hemi_vec, wo) * M_PIf;
			indirect = prd.result * f_d / max(1e-6,prob); //Cosine cancels out
			prd_radiance.seed = prd.seed;
		}
		else
			prd_radiance.seed = t;

		prd_radiance.result = emission + direct + env + indirect;
		optix_print("Glossy (Bounce: %d) Env: %f %f %f, Dir: %f %f %f, Ind: %f %f %f\n", prd_radiance.depth, env.x, env.y, env.z, direct.x, direct.y, direct.z, indirect.x, indirect.y, indirect.z);
	}
	else
	{
		prd_radiance.result = make_float3(0.0f);
	}
	
}