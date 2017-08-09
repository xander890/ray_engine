#include <structs.h>
#include <device_common_data.h>
// Standard ray variables

// Variables for shading
rtDeclareVariable(optix::float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::float3, texcoord, attribute texcoord, ); 

rtDeclareVariable(int, gi, , ); 
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

rtTextureSampler<optix::float4, 2> diffuse_map; 

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { rtTerminateRay(); }

// Closest hit program for ambient light = illumination model 0
RT_PROGRAM void shade() 
{ 
  optix::float3 kd = make_float3(tex2D(diffuse_map, texcoord.x, texcoord.y));
  prd_radiance.result = kd; 
}
