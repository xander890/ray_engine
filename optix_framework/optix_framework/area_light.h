#pragma once
#include <optixu/optixu_vector_types.h>

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

