#ifndef cgla_optix_type_converter_h__
#define cgla_optix_type_converter_h__

#include "optix_math.h"
#include <optix_world.h>
#include <vector>
#include <CGLA/Vec3f.h>
#include <CGLA/Vec2f.h>

inline std::vector<CGLA::Vec3f>* cast(float3 * orig)
{
	return reinterpret_cast<std::vector<CGLA::Vec3f>*>(orig);
}

inline std::vector<CGLA::Vec2f>* cast(float2 * orig)
{
	return reinterpret_cast<std::vector<CGLA::Vec2f>*>(orig);
}
inline CGLA::Vec3f cast(float3 orig)
{
	return CGLA::Vec3f(orig.x,orig.y,orig.z);
}

inline CGLA::Vec2f cast(float2 orig)
{
	return CGLA::Vec2f(orig.x, orig.y);
}
#endif // cgla_optix_type_converter_h__
