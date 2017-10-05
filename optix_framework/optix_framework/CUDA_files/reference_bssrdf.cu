// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <photon_trace_reference_bssrdf.h>

using namespace optix;
rtBuffer<float, 2> resulting_flux;

// Window variables

// Assumes an infinite plane with normal (0,0,1)
__forceinline__ __device__ void infinite_plane_scatter_searchlight(const optix::float3& wi, const float incident_power, const float r, const float theta_s, const float n1_over_n2, const float albedo, const float extinction, const float g, optix::uint & t)
{
	const size_t2 bins = resulting_flux.size();
	// Geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
	const optix::float3 no = ni;

	float3 n = (xo - xi);
	optix_print("wi: %f %f %f, xo-xi: %f %f %f", wi.x, wi.y, wi.z, n.x, n.y, n.z);

	// Refraction
	const float cos_theta_i = optix::max(optix::dot(wi, ni), 0.0f);
	const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
	const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
	const float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
	const optix::float3 w12 = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;
	float flux_t = incident_power;
	optix::float3 xp = xi;
	optix::float3 wp = w12;

	if (!scatter_photon(xp, wp, flux_t, resulting_flux, xo,  n1_over_n2, albedo, extinction, g, t))
	{
		optix_print("Max iterations reached. Distance %f (%f mfps)\n", length(xp - xo), length(xp - xo) / r);
	}
}

#define MILK 0
#define WAX 1
#define A 2
#define B 3
#define C 4
#define D 5
#define E 6
#define MATERIAL B

RT_PROGRAM void reference_bssrdf_camera()
{
	uint idx = launch_index.x;
	optix::uint t = tea<16>(idx, frame + 38);

	const float incident_power = 1.0f;

#if MATERIAL==MILK
	const float theta_i = 20.0f;
	const float sigma_a = 0.0007f;
	const float sigma_s = 1.165;
	const float g = 0.7f;
	const float n1_over_n2 = 1.0f / 1.35f;
	const float r = 1.5f;
	const float theta_s = 0;
	const float albedo = sigma_s / (sigma_s + sigma_a);
	const float extinction = sigma_s + sigma_a;
#elif MATERIAL==WAX
	const float theta_i = 20.0f;
	const float sigma_a = 0.5f;
	const float sigma_s = 1.0f;
	const float g = 0.0f;
	const float n1_over_n2 = 1.0f / 1.4f;
	const float r = 0.5f;
	const float theta_s = 90;
	const float albedo = sigma_s / (sigma_s + sigma_a);
	const float extinction = sigma_s + sigma_a;
#elif MATERIAL==A
	const float theta_i = 30.0f;
	const float albedo = 0.6f;
	const float extinction = 1.0f;
	const float g = 0.0f;
	const float n1_over_n2 = 1.0f / 1.3f;
	const float r = 4.0f;
	const float theta_s = 0;
#elif MATERIAL==B
	const float theta_i = 60.0f;
	const float theta_s = 60;
	const float r = 0.8f;
	const float albedo = 0.99f;
	const float extinction = 1.0f;
	const float g = -0.3f;
	const float n1_over_n2 = 1.0f / 1.4f;
#elif MATERIAL==C
	const float theta_i = 70.0f;
	const float theta_s = 60;
	const float r = 1.0f;
	const float albedo = 0.3f;
	const float extinction = 1.0f;
	const float g = 0.9f;
	const float n1_over_n2 = 1.0f / 1.4f;
#elif MATERIAL==D
	const float theta_i = 0.0f;
	const float theta_s = 105.0f;
	const float r = 4.0f;
	const float albedo = 0.5f;
	const float extinction = 1.0f;
	const float g = 0.0f;
	const float n1_over_n2 = 1.0f / 1.2f;
#elif MATERIAL==E
	const float theta_i = 80.0f;
	const float theta_s = 165.0f;
	const float r = 2.0f;
	const float albedo = 0.8f;
	const float extinction = 1.0f;
	const float g = -0.3f;
	const float n1_over_n2 = 1.0f / 1.3f;
#endif
	const float theta_i_rad = deg2rad(theta_i);
	const float theta_s_rad = deg2rad(-theta_s);
	const optix::float3 wi = normalize(optix::make_float3(-sinf(theta_i_rad), 0, cosf(theta_i_rad)));
	infinite_plane_scatter_searchlight(wi, incident_power, r, theta_s_rad, n1_over_n2, albedo, extinction, g, t);
}

