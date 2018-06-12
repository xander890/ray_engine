#include <device_common_data.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>
#include <math_helpers.h>
using namespace optix;

// Window variables
rtBuffer<float4, 2> output_buffer;
rtBuffer<uchar4, 2> tonemap_output_buffer;

rtDeclareVariable(float, tonemap_multiplier,,) = 1.0f;
rtDeclareVariable(float, tonemap_exponent,,) = 1.8f;

_fn unsigned char to_char(float f)
{
	return (unsigned char)(f * 255.0f);
}

RT_PROGRAM void tonemap_camera()
{
	float4 c = output_buffer[launch_index]; //*
	const float exponent = 1.0f / tonemap_exponent;
	float4 output_color = make_float4(fpowf(tonemap_multiplier*make_float3(c), exponent), c.w);
	output_color = max(min(output_color, make_float4(1)),make_float4(0));

	// GL_BGRA
	tonemap_output_buffer[launch_index] = make_uchar4(to_char(output_color.z), to_char(output_color.y), to_char(output_color.x), to_char(output_color.w));
}
