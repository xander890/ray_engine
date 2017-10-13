
// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <photon_trace_reference_bssrdf.h>
#include <md5.h>
#include <material.h>
#include <bssrdf.h>
using namespace optix;

rtDeclareVariable(BufPtr2D<float>, planar_resulting_flux, , );
rtDeclareVariable(BufPtr2D<float>, planar_resulting_flux_intermediate, , );

rtDeclareVariable(unsigned int, maximum_iterations, , );
rtDeclareVariable(unsigned int, ref_frame_number, , );
rtDeclareVariable(unsigned int, reference_bssrdf_samples_per_frame, , );
// Window variables

rtDeclareVariable(float, reference_bssrdf_theta_i, , );
rtDeclareVariable(float, reference_bssrdf_theta_s, , );
rtDeclareVariable(float, reference_bssrdf_radius, , );
rtDeclareVariable(BufPtr<ScatteringMaterialProperties>, planar_bssrdf_material_params, , );
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );

//#define USE_HARDCODED_MATERIALS

RT_PROGRAM void reference_bssrdf_camera()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);
	float2 angles = get_normalized_hemisphere_buffer_angles(uv.y, uv.x);

	float3 wo = optix::make_float3(sinf(angles.y) * cosf(angles.x), sinf(angles.y) * sinf(angles.x), cosf(angles.y));

	float theta_i; float r; float theta_s; float albedo; float extinction; float g; float n2_over_n1;

#ifdef USE_HARDCODED_MATERIALS
	get_default_material(theta_i, r, theta_s, albedo, extinction, g, n2_over_n1);
#else
	theta_i = reference_bssrdf_theta_i;
	theta_s = reference_bssrdf_theta_s;
	r = reference_bssrdf_radius;
	n2_over_n1 = reference_bssrdf_rel_ior;
	albedo = planar_bssrdf_material_params->albedo.x;
	extinction = planar_bssrdf_material_params->extinction.x;
	g = planar_bssrdf_material_params->meancosine.x;
#endif

	const float theta_i_rad = deg2rad(theta_i);
	const float theta_s_rad = deg2rad(-theta_s);
	const optix::float3 wi = normalize(optix::make_float3(-sinf(theta_i_rad), 0, cosf(theta_i_rad)));

	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
	const optix::float3 no = ni;

	const float n1_over_n2 = 1.0f / n2_over_n1;
	const float cos_theta_i = optix::max(optix::dot(wi, ni), 0.0f);
	const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
	const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
	const float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
	const optix::float3 w12 = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;
	float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_t_i, n1_over_n2);

	const float cos_theta_o = optix::dot(wo, no);
	const float cos_theta_o_sqr = cos_theta_o*cos_theta_o;
	const float sin_theta_to_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o_sqr);
	const float cos_theta_t_o = sqrt(1.0f - sin_theta_to_sqr);
	const optix::float3 w21 = no*cos_theta_t_o - n1_over_n2*(cos_theta_o*no - wo);
	float T21 = 1.0f - fresnel_R(cos_theta_o, cos_theta_t_o, n1_over_n2);

	optix::float3 S = T12 * bssrdf(xi, ni, w12, xo, no, w21, *planar_bssrdf_material_params) * T21;
	planar_resulting_flux_intermediate[launch_index] = S.x;
}

RT_PROGRAM void post_process_bssrdf()
{
	planar_resulting_flux[launch_index] = planar_resulting_flux_intermediate[launch_index];
}
