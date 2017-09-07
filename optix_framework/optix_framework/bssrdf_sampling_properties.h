#pragma once
#include <optix_world.h>
#include <host_device_common.h>

#define BSSRDF_SAMPLING_CAMERA_BASED_MERTENS 0
#define BSSRDF_SAMPLING_NORMAL_BASED_HERY 1
#define BSSRDF_SAMPLING_MIS_KING 2
#define BSSRDF_SAMPLING_METHODS_COUNT 3

#define BSSRDF_SHADERS_SHOW_ALL 0
#define BSSRDF_SHADERS_SHOW_REFRACTION 1
#define BSSRDF_SHADERS_SHOW_REFLECTION 2
#define BSSRDF_SHADERS_SHOW_COUNT 3

struct BSSRDFSamplingProperties
{
	int sampling_method				DEFAULT(BSSRDF_SAMPLING_CAMERA_BASED_MERTENS);
	int correct_camera				DEFAULT(1);
	float R_max						DEFAULT(1.0f);
	optix::float3 mis_weights		DEFAULT(optix::make_float3(0.5f, 0.25f, 0.25f));
	int show_mode					DEFAULT(BSSRDF_SHADERS_SHOW_ALL);
};