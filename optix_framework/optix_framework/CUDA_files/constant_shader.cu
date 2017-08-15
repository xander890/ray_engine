#include <structs.h>
#include <device_common_data.h>
#include <material.h>
// Standard ray variables

// Variables for shading
rtDeclareVariable(optix::float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::float3, texcoord, attribute texcoord, ); 


rtDeclareVariable(MaterialDataCommon, material, , );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { rtTerminateRay(); }

// Closest hit program for ambient light = illumination model 0
RT_PROGRAM void shade() 
{ 
  optix::float3 kd = make_float3(rtTex2D<float4>(material.diffuse_map, texcoord.x, texcoord.y));
  prd_radiance.result = kd; 
}
