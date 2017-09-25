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
	int sampling_method				DEFAULT(BSSRDF_SAMPLING_NORMAL_BASED_HERY);
	int correct_camera				DEFAULT(1);
	float d_max						DEFAULT(1.0f);
	float dot_no_ni_min				DEFAULT(0.001f);
	optix::float3 mis_weights		DEFAULT(optix::make_float3(0.5f, 0.25f, 0.25f));
	int show_mode					DEFAULT(BSSRDF_SHADERS_SHOW_ALL);
	optix::float4 mis_weights_cdf	DEFAULT(optix::make_float4(0.0f, 0.5f, 0.75f, 1.0f));
	int king_full_probe				DEFAULT(1);
};