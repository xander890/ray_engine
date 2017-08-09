#pragma once
#include <optix_world.h>

struct CameraData
{
	optix::float3 eye;
	optix::float3 U;
	optix::float3 V;
	optix::float3 W;
	optix::uint4 render_bounds;
	optix::uint4 rendering_rectangle;
	optix::uint2 camera_size;
	optix::uint downsampling;
	optix::Matrix3x3 inv_calibration_matrix;
};