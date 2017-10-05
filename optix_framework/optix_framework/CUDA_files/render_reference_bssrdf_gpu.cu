#include <device_common_data.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>
#include <math_helpers.h>
#include <colormap.h>
using namespace optix;

// Window variables
rtBuffer<float4, 2> output_buffer;
rtBuffer<float, 2> resulting_flux;
rtBuffer<int, 1>  photon_counter;
rtDeclareVariable(float, reference_scale_multiplier, , );
rtDeclareVariable(unsigned int, show_false_colors, , );

RT_PROGRAM void render_ref()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);
	float2 ip = uv * 2 - make_float2(1); // [-1, 1], this is xd, yd

	// Inverting the projection in the paper:
	float phi_o = atan2f(ip.y, ip.x);
	float l = length(ip);
	float theta_o_normalized = l; // This is |theta_o|, but we remember it is positive.
	float phi_o_normalized = normalize_angle(phi_o) / (2.0f * M_PIf);

	// Normalizing
	float2 coords = make_float2(phi_o_normalized, theta_o_normalized);
	uint2 coords_idx = make_uint2(coords * make_float2(resulting_flux.size()));
	if (l >= 1)
		output_buffer[launch_index] = make_float4(0);
	else
	{
		if (show_false_colors == 1)
			output_buffer[launch_index] = make_float4(jet(reference_scale_multiplier * resulting_flux[coords_idx] / ((float)photon_counter[0])), 1);
		else
			output_buffer[launch_index] = make_float4(reference_scale_multiplier * resulting_flux[coords_idx] / ((float)photon_counter[0]));

	}

}
