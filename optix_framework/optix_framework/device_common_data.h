#ifndef light_common_h__
#define light_common_h__
#include "optix_common.h"
#include "host_device_common.h"
#include "structs.h"
#include "structs_device.h"

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

static __device__ __inline__ void shadow_hit(PerRayData_shadow & shadow_payload, float3 & emission)
{
	if (!(emission.x + emission.y + emission.z > 0.0f))
	{
		shadow_payload.attenuation = 0.0f;
	}

	rtTerminateRay();
}

#ifdef NDEBUG
#define optix_print 
#define optix_assert(x) 
#else
#define optix_print rtPrintf
#define optix_assert(x) if(!(x)) optix_print("Assertion " #x " failed. File: %s, Line %d\n", __FILE__, __LINE__); 
#endif

//#define TEST_SAMPLING
#define TEST_SAMPLING_W 0.01f

#endif // light_common_h__
