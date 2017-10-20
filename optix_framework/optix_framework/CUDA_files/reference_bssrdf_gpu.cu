// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <photon_trace_reference_bssrdf.h>
#include <md5.h>
#include <material.h>
#include <photon_trace_structs.h>
using namespace optix;

rtDeclareVariable(BufPtr2D<float>, reference_resulting_flux, , );
rtDeclareVariable(BufPtr2D<float>, reference_resulting_flux_intermediate, , );

rtDeclareVariable(BufPtr1D<PhotonSample>, photon_buffer, , );
rtDeclareVariable(BufPtr1D<int>, photon_counter, , );


rtDeclareVariable(unsigned int, maximum_iterations, , ) = 1e5;
rtDeclareVariable(unsigned int, batch_iterations, , ) = 1e3;
rtDeclareVariable(unsigned int, ref_frame_number, , ) = 1e5;
rtDeclareVariable(unsigned int, reference_bssrdf_samples_per_frame, , );
// Window variables

rtDeclareVariable(float, reference_bssrdf_theta_i, , );
rtDeclareVariable(float, reference_bssrdf_theta_s, , );
rtDeclareVariable(float, reference_bssrdf_radius, , );
rtDeclareVariable(BufPtr<ScatteringMaterialProperties>, reference_bssrdf_material_params, , );
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );

//#define USE_HARDCODED_MATERIALS

RT_PROGRAM void reference_bssrdf_gpu()
{
	optix_print("Welcome.\n"); 
	uint idx = launch_index.x;
	optix::uint t = ref_frame_number * launch_dim.x + idx;
	hash(t);

	const float incident_power = 1.0f; 
	float theta_i; float r; float theta_s; float albedo; float extinction; float g; float n2_over_n1;

	theta_i = reference_bssrdf_theta_i; 
	theta_s = reference_bssrdf_theta_s;
	r = reference_bssrdf_radius;
	n2_over_n1 = reference_bssrdf_rel_ior;
	albedo = reference_bssrdf_material_params->albedo.x;
	extinction = reference_bssrdf_material_params->extinction.x;
	g = reference_bssrdf_material_params->meancosine.x;

	const float theta_i_rad = theta_i;
	const float theta_s_rad = -theta_s;
	const optix::float3 wi = normalize(optix::make_float3(-sinf(theta_i_rad), 0, cosf(theta_i_rad)));

	// Geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), -sin(theta_s), 0);
	const optix::float3 no = ni;

	PhotonSample p = photon_buffer[idx];

	if (p.status == PHOTON_STATUS_NEW)
	{
		optix_print("New photon.\n");
		p.t = ref_frame_number * launch_dim.x + idx;
		hash(p.t);
		// Refraction      
		const float n1_over_n2 = 1.0f / n2_over_n1;
		optix::float3 w12;
		refract(wi, ni, n1_over_n2, w12);

		p.flux = incident_power;
		p.xp = xi; 
		p.wp = w12;  
		p.i = 0;
		p.status = PHOTON_STATUS_SCATTERING; 
		atomicAdd(&photon_counter[ref_frame_number], 1); 
	}

	if (scatter_photon(p.xp, p.wp, p.flux, reference_resulting_flux_intermediate, xo, n2_over_n1, albedo, extinction, g, p.t, p.i, batch_iterations))
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

RT_PROGRAM void reference_bssrdf_gpu_post()
{
	unsigned int photons = 0;
	for (int i = 0; i < photon_counter.buf.size(); i++)
	{
		photons += photon_counter[i];
	}
	reference_resulting_flux[launch_index] = reference_resulting_flux_intermediate[launch_index] / photons;
}



