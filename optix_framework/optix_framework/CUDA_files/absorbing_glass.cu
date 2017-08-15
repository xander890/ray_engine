// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011


#include <device_common_data.h>
#include <structs.h>
#include <optical_helper.h>
#include <random.h>
#include <ray_trace_helpers.h>
#include <material.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

rtDeclareVariable(float3, ior_complex_real_sq, , ); 
rtDeclareVariable(float3, ior_complex_imag_sq, , ); 
rtDeclareVariable(MaterialDataCommon, material, , );

// Recursive ray tracing variables
rtDeclareVariable(int, max_splits, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { 
    float3 emission = make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
    shadow_hit(prd_shadow, emission);
}

__forceinline__ __device__ void absorbing_glass()
{
  float3 hit_pos = ray.origin + t_hit * ray.direction;
  float3 normal = rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal);
  float3 ffnormal = faceforward(normal, -ray.direction, normal);

  if(prd_radiance.depth < max_depth)
  {
		Ray reflected_ray, refracted_ray;
		float R, cos_theta;
		get_glass_rays(ray, material.ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);

		float xi = rnd(prd_radiance.seed);
		
		if (xi < R)
		{
			PerRayData_radiance prd_new;
			prd_new.depth = prd_radiance.depth + 1;
			prd_new.colorband = prd_radiance.colorband;
			prd_new.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
			prd_new.seed = prd_radiance.seed;
			rtTrace(top_object, reflected_ray, prd_new);
			prd_radiance.seed = prd_new.seed;
			prd_radiance.result = prd_new.result;
		}
		else
		{
			// Ray gets absorbed, no refraction.
			prd_radiance.result = make_float3(0.0f);
		}
  }
  else
  {
    prd_radiance.result = make_float3(0.0f);
  }
}

RT_PROGRAM void shade() { absorbing_glass(); }
RT_PROGRAM void shade_path_tracing() { absorbing_glass(); }