#ifndef light_common_h__
#define light_common_h__
#include "optix_device.h"
#include "host_device_common.h"
#include "structs.h"

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(rtObject, top_object, , );

// Ray generation variables
rtDeclareVariable(optix::uint, frame, , );
rtDeclareVariable(optix::uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(optix::uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(optix::uint2, debug_index, , );
// Exception and debugging variables
rtDeclareVariable(optix::float3, bad_color, , );

// Current ray information
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// Shadow variables

rtDeclareVariable(int, max_depth, , );


_fn void shadow_hit(PerRayData_shadow & shadow_payload, optix::float3 & emission)
{
	if (!(emission.x + emission.y + emission.z > 0.0f))
	{
		shadow_payload.attenuation = 0.0f;
	}

	rtTerminateRay();
}

using optix::dot;
using optix::normalize;
using optix::length;
using optix::cross;

#endif // light_common_h__
