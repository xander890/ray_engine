// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>

#include <material_device.h>
// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() {
    const MaterialDataCommon & material = get_material();
    float3 emission = make_float3(optix::rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
	 shadow_hit(prd_shadow,emission);
}

// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void shade() 
{ 
	float3 hit_pos = ray.origin + t_hit * ray.direction;
	float3 normal = rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal); 
	float3 ffnormal = faceforward(normal, -ray.direction, normal); 

	if(prd_radiance.depth < max_depth)
	{
		PerRayData_radiance prd_new;
		prd_new.depth = prd_radiance.depth + 1;

		prd_new.colorband = prd_radiance.colorband;
		prd_new.seed = prd_radiance.seed;
		prd_new.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
		float3 new_dir = reflect(ray.direction, normal);
		prd_new.result = make_float3(0.0f);
		optix::Ray new_ray(hit_pos, new_dir, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(top_object, new_ray, prd_new);
		prd_radiance.result = prd_new.result; 
		prd_radiance.seed = prd_new.seed;
		
	}
	else
	{
		prd_radiance.result = make_float3(0.0f);
	}

}
