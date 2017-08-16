#include <structs.h>
#include <device_common_data.h>
#include <material_device.h>
// Standard ray variables

// Variables for shading
rtDeclareVariable(optix::float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::float3, texcoord, attribute texcoord, ); 



rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { rtTerminateRay(); }

// Closest hit program for ambient light = illumination model 0
RT_PROGRAM void shade() 
{ 
    float3 hit_pos = ray.origin + t_hit * ray.direction;
    int material_idx = (hit_pos.z > 0) ? 0 : 1;
    MaterialDataCommon& mat = material_buffer[material_idx];
    float3 k_d = make_float3(rtTex2D<float4>(mat.diffuse_map, texcoord.x, texcoord.y));
  prd_radiance.result = k_d; 
}
