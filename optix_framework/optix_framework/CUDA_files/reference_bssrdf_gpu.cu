// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <photon_trace_structs.h>
#include <photon_trace_reference_bssrdf.h>

using namespace optix;
rtBuffer<float, 2> resulting_flux;
rtBuffer<PhotonSample, 1> photon_buffer;
rtBuffer<int, 1>  photon_counter;
rtDeclareVariable(unsigned int, maximum_iterations, , );
rtDeclareVariable(unsigned int, batch_iterations, , );
rtDeclareVariable(unsigned int, ref_frame_number, , );


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

	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s_rad), sin(theta_s_rad), 0);
	const optix::float3 no = ni;
	const float incident_power = 1.0f;

	PhotonSample p = photon_buffer[idx];
	if (p.status == PHOTON_STATUS_NEW)
	{
		optix_print("New photon.\n");
		p.t = tea<16>(idx, ref_frame_number);
		const float cos_theta_i = optix::max(optix::dot(wi, ni), 0.0f);
		const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
		const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
		const float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
		const optix::float3 w12 = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;
		p.flux = incident_power;
		p.xp = xi;
		p.wp = w12;
		p.i = 0;
		p.status = PHOTON_STATUS_SCATTERING;
		atomicAdd(&photon_counter[0], 1);
	}

	if (scatter_photon(p.xp, p.wp, p.flux, resulting_flux, xo, n1_over_n2, albedo, extinction, g, p.t, p.i, batch_iterations))
	{
		photon_buffer[idx].status = PHOTON_STATUS_NEW;
	}
	else
	{
		p.i += batch_iterations;
		if (p.i >= maximum_iterations)
		{
			optix_print("Max iterations reached. Distance %f (%f mfps)\n", length(p.xp - xo), length(p.xp - xo) / r);
			photon_buffer[idx].status = PHOTON_STATUS_NEW;
		}
		else
		{
			optix_print("Continuing %d.\n", p.i);
			photon_buffer[idx] = p;
		}
	}
}

