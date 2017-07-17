#ifndef light_common_h__
#define light_common_h__
#include "optix_common.h"
#include "structs.h"
#include "structs_device.h"

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(rtObject, top_object, , );

// Shadow variables

// Recursive ray tracing variables
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(unsigned int, dummy_ray_type, , );

static __device__ __inline__ void shadow_hit(PerRayData_shadow & shadow_payload, float3 & emission)
{
	if (!(emission.x + emission.y + emission.z > 0.0f))
	{
		shadow_payload.attenuation = 0.0f;
	}

	rtTerminateRay();
}

#define optix_print rtPrintf

#endif // light_common_h__
