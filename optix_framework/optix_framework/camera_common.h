#pragma once
#include "host_device_common.h"

/*
 * Struct to store different camera parameters.
 */
struct CameraData
{
	optix::float3 eye;					// Camera mEye position
	optix::float3 U;					// Camera horizonthal vector, scaled with correct focal length and fov.
	optix::float3 V;					// Camera vertical vector, scaled with correct focal length and fov.
	optix::float3 W;					// Camera front vector, scaled with correct focal length.
	optix::uint downsampling;			// Downsampling of the screen, to render at lower resolution.
	optix::Matrix3x3 view_matrix;		// View rotation matrix.
	optix::Matrix3x3 inv_view_matrix;	// Inverse of view rotation matrix.
};