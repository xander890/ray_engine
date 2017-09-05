// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>
#include <material_device.h>
#include <ray_trace_helpers.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 






// Russian roulette variables
rtDeclareVariable(int, max_splits, , );


// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { 
    const MaterialDataCommon & material = get_material();
    float3 emission = make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
 shadow_hit(prd_shadow, emission);
}


// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void shade() 
{ 
	float3 color = make_float3(0.0f);
    const MaterialDataCommon & material = get_material();
	
  if(prd_radiance.depth < max_depth)
  {
		float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
		float3 ffnormal = faceforward(normal, -ray.direction, normal);
		float3 hit_pos = ray.origin + t_hit * ray.direction;


		PerRayData_radiance prd_refract = prepare_new_pt_payload(prd_radiance);

		PerRayData_radiance prd_refl = prepare_new_pt_payload(prd_radiance);

		Ray reflected_ray, refracted_ray;
		float R, cos_theta;
		get_glass_rays(ray, material.ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);

		rtTrace(top_object, reflected_ray, prd_refl);
		prd_refract.seed = prd_refl.seed;

		color += R * prd_refl.result;
		rtTrace(top_object, refracted_ray, prd_refract);
		color += (1-R) * prd_refract.result;
		prd_radiance.seed = prd_refract.seed;
		
	}
	prd_radiance.result = color; 
}

RT_PROGRAM void shade_path_tracing(void)
{
  float3 color = make_float3(0.0f);
  const MaterialDataCommon & material = get_material();

  if(prd_radiance.depth < max_depth)
  {
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 hit_pos = ray.origin + t_hit*ray.direction;
    hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);

	PerRayData_radiance prd_new_ray = prepare_new_pt_payload(prd_radiance);
    Ray reflected_ray, refracted_ray;
    float R, cos_theta;
    get_glass_rays(ray, material.ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);

	// Russian roulette with absorption if inside
	float3 beam_T = make_float3(1.0f);
	if (cos_theta < 0.0f)
	{
		beam_T = expf(-t_hit*material.absorption);
		float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
		if (rnd(prd_new_ray.seed) >= prob)
		{
			prd_radiance.result = make_float3(0);
			return;
		}
		beam_T /= max(10e-6,prob);
	}

    float xi = rnd(prd_new_ray.seed);
	optix::Ray & ray = (xi < R) ? reflected_ray : refracted_ray;
	rtTrace(top_object, ray, prd_new_ray);
	color = prd_new_ray.result;    
	color *= beam_T;
    prd_radiance.seed = prd_new_ray.seed;

	optix_print("Glass - (Bounce: %d) Color: %f %f %f (R: %f) Costheta: %f\n", prd_radiance.depth, color.x, color.y, color.z, R, cos_theta);
  }
  prd_radiance.result = color;
}
