#pragma once
#include "host_device_common.h"
/*
 * This struct represents one of the elements of an area light in triangle shape. Light contains three triangle vertices, the normal, the area of the light and, the stored radiance.
 */
struct TriangleLight
{
	optix::float3 v1, v2, v3;
	optix::float3 normal;
	optix::float3 emission;
	float area;
};

struct SingularLightData
{
	optix::float3 direction   DEFAULT(optix::make_float3(0,-1,0));
	LightType::Type type      DEFAULT(LightType::DIRECTIONAL);
	optix::float3 emission    DEFAULT(optix::make_float3(1,1,1));
	int casts_shadow          DEFAULT(1);
};


