// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(uint, ray_traced_reflection, ,);
rtDeclareVariable(float, ior, , );
rtDeclareVariable(float3, ior_complex_real_sq, , );
rtDeclareVariable(float3, ior_complex_imag_sq, , );
rtDeclareVariable(float3, absorption, , );

// Material properties (corresponding to OBJ mtl params)
rtTextureSampler<float4, 2> ambient_map;
rtTextureSampler<float4, 2> diffuse_map; 
rtTextureSampler<float4, 2> specular_map; 

rtDeclareVariable(int, max_depth, , );

// Russian roulette variables
rtDeclareVariable(int, max_splits, , );


// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { 
	 float3 emission = make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));

	 shadow_hit(prd_shadow, emission);
}


// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void shade() 
{ 
	float3 color = make_float3(0.0f);

	
  if(prd_radiance.depth < max_depth)
  {
		float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
		float3 ffnormal = faceforward(normal, -ray.direction, normal);
		float3 hit_pos = ray.origin + t_hit * ray.direction;


		PerRayData_radiance prd_refract;
		prd_refract.depth = prd_radiance.depth + 1;
		prd_refract.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
		prd_refract.colorband = prd_radiance.colorband;

		PerRayData_radiance prd_refl;
		prd_refl.depth = prd_radiance.depth + 1;
		prd_refl.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
		prd_refl.colorband = prd_radiance.colorband;

		Ray reflected_ray, refracted_ray;
		float R, cos_theta;
		get_glass_rays(ray, ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);

		rtTrace(top_object, reflected_ray, prd_refl);
		color += R * prd_refl.result;
		rtTrace(top_object, refracted_ray, prd_refract);
		color += (1-R) * prd_refract.result;
		
	}
	prd_radiance.result = color; 
}

RT_PROGRAM void shade_rr(void)
{
	float3 hit_pos = ray.origin + t_hit * ray.direction;
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(normal, -ray.direction, normal);
	float3 color = make_float3(0.0f);


	if (prd_radiance.depth < max_depth)
	{
	    PerRayData_radiance prd_refract;
		PerRayData_radiance prd_refl;

		Ray reflected_ray, refracted_ray;
		float R, cos_theta;
		get_glass_rays(ray, ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);

		uint t = prd_radiance.seed;
		float random = rnd(t);
		bool split = prd_radiance.depth < max_splits;
		if (split || random <= R)
		{
			prd_refl.depth = prd_radiance.depth + 1;
			prd_refl.colorband = prd_radiance.colorband;
			prd_refl.seed = t;
			prd_refl.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
			rtTrace(top_object, reflected_ray, prd_refl);
			R = split ? R : 1;
			color += R * prd_refl.result;
			prd_radiance.seed = prd_refl.seed;
		}
		
		if (split || random > R)
		{
			prd_refract.seed = t;
			prd_refract.depth = prd_radiance.depth + 1;
			prd_refract.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
			prd_refract.colorband = prd_radiance.colorband;
			rtTrace(top_object, refracted_ray, prd_refract);
			R = split ? R : 0;
			color += (1-R) * prd_refract.result;
			prd_radiance.seed = prd_refract.seed;
		}
	}

	prd_radiance.result = color;
	
}

RT_PROGRAM void shade_path_tracing(void)
{
  float3 color = make_float3(0.0f);
  optix_print("Glass Hit\n");
  if(prd_radiance.depth < max_depth)
  {
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_pos = ray.origin + t_hit*ray.direction;
    hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);
    uint t = prd_radiance.seed;

    PerRayData_radiance prd_new_ray;
    prd_new_ray.depth = prd_radiance.depth + 1;
    prd_new_ray.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
	prd_new_ray.colorband = prd_radiance.colorband;

    Ray reflected_ray, refracted_ray;
    float R, cos_theta;
    get_glass_rays(ray, ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);


	// Russian roulette with absorption if inside
	float3 beam_T = make_float3(1.0f);
	if (cos_theta < 0.0f)
	{
		beam_T = expf(-t_hit*absorption);
		float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
		if (rnd(t) >= prob)
		{
			prd_radiance.result = make_float3(0);
			return;
		}
		beam_T /= max(10e-6,prob);
	}

    if(prd_radiance.depth < max_splits)
    {
      prd_new_ray.seed = t;
      rtTrace(top_object, reflected_ray, prd_new_ray);
      color = R*prd_new_ray.result;
      prd_new_ray.depth = prd_radiance.depth + 1;
	  prd_new_ray.colorband = prd_radiance.colorband;
      prd_new_ray.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
      rtTrace(top_object, refracted_ray, prd_new_ray);
      color += (1.0f - R)*prd_new_ray.result;
    }
    else
    {
      float xi = rnd(t);
      prd_new_ray.seed = t;
	  optix::Ray & ray = (xi < R) ? reflected_ray : refracted_ray;
	  rtTrace(top_object, ray, prd_new_ray);
	  color = prd_new_ray.result;

	}
    
	color *= beam_T;
    prd_radiance.seed = prd_new_ray.seed;

	optix_print("Glass - (Bounce: %d) Color: %f %f %f (R: %f) Costheta: %f\n", prd_radiance.depth, color.x, color.y, color.z, R, cos_theta);
  }
  prd_radiance.result = color;
}
