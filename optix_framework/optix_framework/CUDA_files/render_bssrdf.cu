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
rtDeclareVariable(unsigned int, show_false_colors, , );
rtDeclareVariable(float, reference_scale_multiplier, , );
rtDeclareVariable(TexPtr, resulting_flux_tex, , );
rtDeclareVariable(int, reference_bssrdf_output_shape, , ) = BSSRDF_OUTPUT_HEMISPHERE;
rtDeclareVariable(float, ior, , );

RT_PROGRAM void render_ref()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);
	float2 ip = uv * 2 - make_float2(1); // [-1, 1], this is xd, yd

	if (reference_bssrdf_output_shape == BSSRDF_OUTPUT_HEMISPHERE)
	{
		// Inverting the projection in the paper:
		float phi_o = atan2f(ip.y, ip.x);
		float l = length(ip);


		if (l >= 1)
		{
			output_buffer[launch_index] = make_float4(0);
		}
		else
		{
			float2 coords = get_normalized_hemisphere_buffer_coordinates(l * M_PIf * 0.5f, phi_o);
			float val = reference_scale_multiplier * optix::rtTex2D<float4>(resulting_flux_tex, coords.x, coords.y).x;
			if (show_false_colors == 1)
				output_buffer[launch_index] = make_float4(jet(val), 1);
			else
				output_buffer[launch_index] = make_float4(val);

		}
	}
	else
	{
		float fresnel_integral = C_phi(ior) * 4 * M_PIf;
		float R21;
		optix::float3 w21; 
		const optix::float3 no = optix::make_float3(0,0,1);
		const optix::float3 wo = no;
		refract(wo, no, 1 / ior , w21, R21);
		float T21 = 1.0f - R21;

		float S_shown = reference_scale_multiplier * fresnel_integral / T21 * optix::rtTex2D<float4>(resulting_flux_tex, uv.x, uv.y).x;

		float t = clamp((log(S_shown + 1.0e-10) / 2.30258509299f + 6.0f) / 6.0f, 0.0f, 1.0f);
		float h = clamp((1.0 - t)*2.0f, 0.0f, 0.65f);

		optix::float4 res = make_float4(S_shown);
		if (show_false_colors)
			res = optix::make_float4(hsv2rgb(h, 1.0, 1.0), 1.0);
		output_buffer[launch_index] = res;
	}
	
}