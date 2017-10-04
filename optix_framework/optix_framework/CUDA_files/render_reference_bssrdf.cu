#include <device_common_data.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>
#include <math_helpers.h>
using namespace optix;

// Window variables
rtBuffer<float4, 2> output_buffer;

RT_PROGRAM void render_ref()
{
	output_buffer[launch_index] = make_float4(1,0,0,1); //*
}
