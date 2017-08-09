

#include <device_common_data.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>

using namespace optix;

// Window variables
rtBuffer<float4, 2> output_buffer;
rtBuffer<float4, 2> tonemap_output_buffer;

rtDeclareVariable(float, tonemap_multiplier,,) = 1.0f;
rtDeclareVariable(float, tonemap_exponent,,) = 1.8f;

rtDeclareVariable(float, comparison_image_weight, , ) = 0;
rtDeclareVariable(int, show_difference_image, , ) = 0;
rtDeclareVariable(int, comparison_texture, , ) = 0;

RT_PROGRAM void tonemap_camera()
{
	float4 c = output_buffer[launch_index]; //*
	const float exponent = 1.0f / tonemap_exponent;
	float4 tm_color = make_float4(fpowf(tonemap_multiplier*make_float3(c), exponent), c.w);



	float4 comp_color = rtTex2D<float4>(comparison_texture, ((float)launch_index.x) / launch_dim.x, ((float)launch_index.y) / launch_dim.y);
	if (show_difference_image == 0)
	{

		if (comparison_image_weight < 0.01f)
		{
			tonemap_output_buffer[launch_index] = tm_color;
		}
		else
		{
			
			tonemap_output_buffer[launch_index] = comp_color * comparison_image_weight + tm_color * (1 - comparison_image_weight);
		}
	}
	else
	{
		tonemap_output_buffer[launch_index] = make_float4(length(comp_color - tm_color));
	}
}
