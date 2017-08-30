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
rtDeclareVariable(PerRayData_depth, prd_radiance, rtPayload, );

// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void depth()
{
    prd_radiance.depth = t_hit;
}