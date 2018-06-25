// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common.h>
#include <color_utils.h>
#include <environment_map.h>
#include "material_device.h"

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { rtTerminateRay(); }

// Closest hit program for drawing shading normals
RT_PROGRAM void shade()
{
    const int material_size = material_buffer.size();
    const int material_index = get_material_index(texcoord);
    prd_radiance.result = hsv2rgb((float)material_index / material_size, 1.0, 1.0);
}