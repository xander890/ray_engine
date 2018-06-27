#pragma once
#include "optix_device.h"
#include "host_device_common.h"
#include "structs.h"
// Top header for .cu shaders implemented on the GPU. Provides a common interface to standard optix variable. Do NOT redeclare them in your shader, simply include this header.

// Scene variables
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

// Maximum ray depth
rtDeclareVariable(int, max_depth, , );

using optix::dot;
using optix::normalize;
using optix::length;
using optix::cross;
