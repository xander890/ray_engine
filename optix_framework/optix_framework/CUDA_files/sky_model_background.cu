// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011


#include "..\structs.h"
#include "..\sky_model.h"

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// Variables for shading
rtDeclareVariable(PerezData, perez_model_data, , );
rtDeclareVariable(float3, sun_position, , );
rtDeclareVariable(float3, up_vector, , );
rtDeclareVariable(float3, sky_factor, , );
rtDeclareVariable(float3, sun_color, , );

// Miss program returning background color
RT_PROGRAM void miss()
{
	float3 v = normalize(ray.direction);
	float3 color = make_float3(0.0f);

	if (prd_radiance.flags & RayFlags::USE_EMISSION)
	{
		color = sky_color(prd_radiance.depth, v, sun_position, up_vector, sky_factor, sun_color, perez_model_data);
	}
	prd_radiance.result = color;
}

// Miss program returning background color
RT_PROGRAM void miss_shadow()
{
	prd_shadow.attenuation = 1.0f;
}
