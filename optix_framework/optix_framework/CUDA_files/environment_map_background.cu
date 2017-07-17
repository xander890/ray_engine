// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>
using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );

rtDeclareVariable(Matrix3x3, lightmap_rotation_matrix, , );
rtDeclareVariable(float3, lightmap_multiplier, , );
// Variables for shading
rtDeclareVariable(float3, bg_color, , );

rtDeclareVariable(int, environment_map_tex_id, , ) = 0;


__device__ __forceinline__ void get_environment_map_color(const float3& direction, float3 & color)
{
	const float2 uv = direction_to_uv_coord_cubemap(direction, lightmap_rotation_matrix);
	color = make_float3(rtTex2D<float4>(environment_map_tex_id, uv.x, uv.y)) * lightmap_multiplier;
}

// Miss program returning background color
RT_PROGRAM void miss()
{
  float3 color = make_float3(0.0f);
  if (prd_radiance.flags & RayFlags::USE_EMISSION)
  {
	get_environment_map_color(ray.direction, color);
  }
  prd_radiance.result = color;
  //prd_radiance.result = make_float3(tex2D(environment_map, ((float)launch_index.x) / launch_dim.x, ((float)launch_index.y) / launch_dim.y)) * lightmap_multiplier;
  optix_print("Ray miss, hit envmap. Returning color %f %f %f\n", color.x, color.y, color.z);
}

// Miss program returning background color
RT_PROGRAM void miss_shadow()
{
	float cos_theta;
	float3 color = make_float3(0.0f);
	get_environment_map_color(ray.direction, color);
	prd_shadow.emission = color;
	prd_shadow.attenuation = 1.0f;
	optix_print("Shadow ray miss, hit envmap. Returning color %f %f %f\n", color.x, color.y, color.z);
}
