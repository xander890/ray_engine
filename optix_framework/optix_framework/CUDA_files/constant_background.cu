// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, bg_color, , );

// Miss program returning background color
RT_PROGRAM void miss()
{
	float3 color = make_float3(0.0f);


	if (prd_radiance.flags & RayFlags::USE_EMISSION)
	{
		color = bg_color;
	}

	prd_radiance.result = color;
	optix_print("Shadow ray miss. Returning color %f %f %f\n", color.x, color.y, color.z);
}

// Miss program returning background color
RT_PROGRAM void miss_shadow()
{
	prd_shadow.attenuation = 1.0f;
	prd_shadow.emission = bg_color;
	optix_print("Shadow ray miss. Returning color %f %f %f\n", bg_color.x, bg_color.y, bg_color.z);
}
