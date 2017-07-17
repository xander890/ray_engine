#ifndef structs_device_h__
#define structs_device_h__

#include <optix_device.h>
struct HitInfo
{
	__device__ HitInfo(optix::float3 & hit_point, optix::float3 & hit_normal) :
		hit_point(hit_point),
		hit_normal(hit_normal)
	{}

	optix::float3 & hit_point;
	optix::float3 & hit_normal;
};
#endif // structs_device_h__
