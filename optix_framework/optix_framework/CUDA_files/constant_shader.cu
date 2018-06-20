#include <structs.h>
#include <device_common.h>
#include <material_device.h>
// Standard ray variables

// Variables for shading
rtDeclareVariable(optix::float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::float2, texcoord, attribute texcoord, );

using optix::rtTex2D;
using optix::make_float3;
using optix::float4;
using optix::float3;

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}

// Closest hit program for ambient light = illumination model 0
RT_PROGRAM void shade() 
{ 
    MaterialDataCommon mat = get_material(texcoord);
    float3 k_d = make_float3(rtTex2D<float4>(mat.diffuse_map, texcoord.x, texcoord.y));	
	prd_radiance.result = k_d;
}
