// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>
#include <material_device.h>
#include <ray_trace_helpers.h>

using namespace optix;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Standard ray variables
rtDeclareVariable(PerRayData_depth, prd_radiance, rtPayload, );

// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void depth()
{
    prd_radiance.depth = t_hit;
}

// Standard ray variables
rtDeclareVariable(PerRayData_normal_depth, prd_attr, rtPayload, );

// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void attribute_closest_hit()
{
	optix_print("Depth ray hit! ID %d\n", mesh_id);
	prd_attr.depth = t_hit;
	prd_attr.normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
}