#include <device_common_data.h>
#include <full_bssrdf_host_device_common.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>
#include <math_helpers.h>
#include <colormap.h>
#include <photon_trace_reference_bssrdf.h>
#include <scattering_properties.h>
using namespace optix;

// Window variables
rtBuffer<float4, 2> output_buffer;
rtDeclareVariable(unsigned int, show_false_colors, , ); 
rtDeclareVariable(float, reference_scale_multiplier, , );
rtDeclareVariable(TexPtr, resulting_flux_tex, , );
rtDeclareVariable(OutputShape::Type, reference_bssrdf_output_shape, , ) = OutputShape::HEMISPHERE;
rtDeclareVariable(int, reference_bssrdf_fresnel_mode, , ) = BSSRDF_RENDER_MODE_FULL_BSSRDF;
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );

__device__ __forceinline__ float convert_to_tex_coordinate(float normalized_buffer_coordinate, unsigned int size)
{
	float sz = static_cast<float>(size);
	float factor = (sz - 1.0f) / sz;
	return 0.5f / sz + normalized_buffer_coordinate * factor;
}

RT_PROGRAM void render_ref()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);
	float2 ip = uv * 2 - make_float2(1); // [-1, 1], this is xd, yd

	if (reference_bssrdf_output_shape == OutputShape::HEMISPHERE)
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
			float theta_o = M_PIf * 0.5f * l;
			float2 coords = get_normalized_hemisphere_buffer_coordinates(phi_o,theta_o);
			optix::uint3 size = optix::rtTexSize(resulting_flux_tex);
			float2 texcoords;
			texcoords.x = convert_to_tex_coordinate(coords.x, size.x);
			texcoords.y = convert_to_tex_coordinate(coords.y, size.y);

			float cos_theta_o = cosf(theta_o);
			float S = reference_scale_multiplier * optix::rtTex2D<float4>(resulting_flux_tex, texcoords.x, texcoords.y).x;
            //float S = reference_scale_multiplier * optix::rtTex2DFetch<float4>(resulting_flux_tex, int(coords.x * size.x), int(coords.y * size.y)).x;
			float T21 = 1.0f - fresnel_R(cos_theta_o, reference_bssrdf_rel_ior);

			float val;
			switch (reference_bssrdf_fresnel_mode)
			{
			case BSSRDF_RENDER_MODE_FRESNEL_OUT_ONLY: val = T21; break;
			case BSSRDF_RENDER_MODE_REMOVE_FRESNEL: val = S / T21; break;
			default:
			case BSSRDF_RENDER_MODE_FULL_BSSRDF: val = S; break;
			}

			if (show_false_colors == 1)
				output_buffer[launch_index] = make_float4(jet(val), 1);
			else
				output_buffer[launch_index] = make_float4(val);
		}
	}
	else
	{
		float fresnel_integral = C_phi(reference_bssrdf_rel_ior) * 4 * M_PIf;
		float R21;
		optix::float3 w21; 
		const optix::float3 no = optix::make_float3(0,0,1);
		const optix::float3 wo = no;
		refract(wo, no, 1 / reference_bssrdf_rel_ior, w21, R21);
		float T21 = 1.0f - R21;

		float S = optix::rtTex2D<float4>(resulting_flux_tex, uv.x, uv.y).x;
		float S_shown = reference_scale_multiplier * fresnel_integral / T21 * S;

		float t = clamp((logf(S_shown + 1.0e-10f) / 2.30258509299f + 6.0f) / 6.0f, 0.0f, 1.0f);
		float h = clamp((1.0f - t)*2.0f, 0.0f, 0.65f);

		optix::float4 res = make_float4(S_shown);
		if (show_false_colors)
		{
			// Jet, but with FD17 Color visualization:
			float Slog = log10(S);
			Slog = (Slog + 7.0f) / 8.0f; // Between -7 and 1
			Slog = clamp(Slog, 0.0f, 1.0f);
			res = optix::make_float4(jet(Slog), 1);
			// Standard matlab Jet
			//res = optix::make_float4(jet(S_shown), 1);
			// Jeppe visualization
			res = optix::make_float4(hsv2rgb(h, 1.0, 1.0), 1.0);
		}
		output_buffer[launch_index] = res;
	}
	
}
