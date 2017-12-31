#include <device_common_data.h>
#include <color_helpers.h>
#include <ray_trace_helpers.h>
#include <environment_map.h>
#include <math_helpers.h>
#include <colormap.h>
#include <scattering_properties.h>
#include <bssrdf.h>
#include "optical_helper.h"

using namespace optix; 

// Window variables  
rtBuffer<float4, 2> output_buffer;

rtDeclareVariable(float, reference_bssrdf_theta_i, , );
rtDeclareVariable(BufPtr<ScatteringMaterialProperties>, planar_bssrdf_material_params, , );
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );
rtDeclareVariable(unsigned int, show_false_colors, , );
rtDeclareVariable(unsigned int, channel_to_show, , );
rtDeclareVariable(float, scale_multiplier, , );

RT_PROGRAM void render()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);
	float2 ip = uv * 2 - make_float2(1); // [-1, 1], this is xd, yd

	ip *= -2;

    BSSRDFGeometry geometry;
    geometry.xi = optix::make_float3(0, 0, 0);
    geometry.ni = optix::make_float3(0, 0, 1);
    geometry.xo = optix::make_float3(ip.x, ip.y, 0);
    geometry.no = optix::make_float3(0, 0, 1);
    geometry.wo = geometry.no;

	const float theta_i_rad = reference_bssrdf_theta_i;
	geometry.wi = normalize(optix::make_float3(sinf(theta_i_rad), 0, cosf(theta_i_rad)));

	optix::float3 w12;
	float R12;  
	refract(geometry.wi, geometry.ni, 1.0f / reference_bssrdf_rel_ior, w12, R12);
	float T12 = 1.0f - R12;
	float fresnel_integral = planar_bssrdf_material_params->C_phi * 4 * M_PIf;
	 
	optix_assert(channel_to_show >= 0 && channel_to_show < 3);

    MaterialDataCommon mat;
    mat.scattering_properties = planar_bssrdf_material_params[0];
    
	float3 S = T12 * fresnel_integral * bssrdf(geometry, 1.0f / reference_bssrdf_rel_ior, mat);
	float S_shown = optix::get_channel(channel_to_show, S) * scale_multiplier;

	float t = optix::clamp((logf(S_shown + 1.0e-10) / 2.30258509299f + 6.0f) / 6.0f, 0.0f, 1.0f);
	float h = optix::clamp((1.0f - t)*2.0f, 0.0f, 0.65f);
	 
	optix::float4 res = make_float4(S_shown);
	if(show_false_colors)
		res = optix::make_float4(hsv2rgb(h, 1.0, 1.0), 1.0);
	output_buffer[launch_index] = res;
	optix_print("%f %f %f", reference_bssrdf_rel_ior, planar_bssrdf_material_params->absorption.x, planar_bssrdf_material_params->scattering.x);

}
 
