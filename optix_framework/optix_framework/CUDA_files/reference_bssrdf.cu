// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <photon_trace_reference_bssrdf.h>
#include <md5.h>
#include <material.h>
using namespace optix;
rtBuffer<float, 2> resulting_flux;
rtDeclareVariable(unsigned int, maximum_iterations, , );
rtDeclareVariable(unsigned int, ref_frame_number, , );
rtDeclareVariable(unsigned int, reference_bssrdf_samples_per_frame, , );
// Window variables

rtDeclareVariable(float, reference_bssrdf_theta_i, , );
rtDeclareVariable(float, reference_bssrdf_theta_s, , );
rtDeclareVariable(float, reference_bssrdf_radius, , );
rtDeclareVariable(float, reference_bssrdf_albedo, , );
rtDeclareVariable(float, reference_bssrdf_extinction, , );
rtDeclareVariable(float, reference_bssrdf_g, , );
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );

//#define USE_HARDCODED_MATERIALS

// Assumes an infinite plane with normal (0,0,1)
__forceinline__ __device__ void infinite_plane_scatter_searchlight(const optix::float3& wi, const float incident_power, const float r, const float theta_s, const float n2_over_n1, const float albedo, const float extinction, const float g, optix::uint & t)
{
	// Geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
	const optix::float3 no = ni;

	// Refraction
	const float n1_over_n2 = 1.0f / n2_over_n1;
	const float cos_theta_i = optix::max(optix::dot(wi, ni), 0.0f);
	const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
	const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
	const float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
	const optix::float3 w12 = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;

	float flux_t = incident_power / ((float)reference_bssrdf_samples_per_frame);
	optix::float3 xp = xi;
	optix::float3 wp = w12;

	if (!scatter_photon(xp, wp, flux_t, resulting_flux, xo,  n2_over_n1, albedo, extinction, g, t, 0, maximum_iterations))
	{
		optix_print("Max iterations reached. Distance %f (%f mfps)\n", length(xp - xo), length(xp - xo) / r);
	}
}

RT_PROGRAM void reference_bssrdf_camera()
{
	uint idx = launch_index.x;
	optix::uint t = ref_frame_number * launch_dim.x + idx;
	hash(t);

	const float incident_power = 1.0f;
	float theta_i; float r; float theta_s; float albedo; float extinction; float g; float n2_over_n1;

#ifdef USE_HARDCODED_MATERIALS
	get_default_material(theta_i, r, theta_s, albedo, extinction, g, n2_over_n1);
#else
	theta_i = reference_bssrdf_theta_i;
	theta_s = reference_bssrdf_theta_s;
	r = reference_bssrdf_radius;
	n2_over_n1 = reference_bssrdf_rel_ior;
	albedo = reference_bssrdf_albedo;
	extinction = reference_bssrdf_extinction;
	g = reference_bssrdf_g;
#endif

	const float theta_i_rad = deg2rad(theta_i);
	const float theta_s_rad = deg2rad(-theta_s);
	const optix::float3 wi = normalize(optix::make_float3(-sinf(theta_i_rad), 0, cosf(theta_i_rad)));
	infinite_plane_scatter_searchlight(wi, incident_power, r, theta_s_rad, n2_over_n1, albedo, extinction, g, t);
}

