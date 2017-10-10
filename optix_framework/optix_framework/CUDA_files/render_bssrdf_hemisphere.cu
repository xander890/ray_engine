#include <device_common_data.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>
#include <math_helpers.h>
#include <colormap.h>
#include <photon_trace_reference_bssrdf.h>
using namespace optix;

// Window variables
rtBuffer<float4, 2> output_buffer;
rtDeclareVariable(unsigned int, reference_bssrdf_samples_per_frame, , );
rtDeclareVariable(unsigned int, show_false_colors, , );
rtDeclareVariable(float, reference_scale_multiplier, , );
rtDeclareVariable(TexPtr, resulting_flux_tex, , );

RT_PROGRAM void render_ref()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);
	float2 ip = uv * 2 - make_float2(1); // [-1, 1], this is xd, yd

	// Inverting the projection in the paper:
	float phi_o = atan2f(ip.y, ip.x);
	float l = length(ip);
	float2 coords = get_normalized_hemisphere_buffer_coordinates(l * M_PIf * 0.5f, phi_o);

	if (l >= 1)
	{
		output_buffer[launch_index] = make_float4(0);
	}
	else
	{
		float val = reference_scale_multiplier * optix::rtTex2D<float4>(resulting_flux_tex, coords.x, coords.y).x;
		
		if(show_false_colors == 1)
			output_buffer[launch_index] = make_float4(jet(val), 1);
		else
			output_buffer[launch_index] = make_float4(val);
	}
}
