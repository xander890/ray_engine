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
// Window variables

rtDeclareVariable(float, reference_bssrdf_theta_i, , );
rtDeclareVariable(float, reference_bssrdf_theta_s, , );
rtDeclareVariable(float, reference_bssrdf_radius, , );
rtDeclareVariable(BufPtr<ScatteringMaterialProperties>, reference_bssrdf_material_params, , );
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );
rtDeclareVariable(int, reference_bssrdf_output_shape, , );

//#define USE_HARDCODED_MATERIALS

RT_PROGRAM void reference_bssrdf_gpu()
{
	optix_print("Welcome.\n"); 
	uint idx = launch_index.x;
	optix::uint t = ref_frame_number * launch_dim.x + idx;
	tea_hash(t);

	const float incident_power = 1.0f; 
	float theta_i; float r; float theta_s; float albedo; float extinction; float g; float n2_over_n1;

	theta_i = reference_bssrdf_theta_i; 
	theta_s = reference_bssrdf_theta_s;
	r = reference_bssrdf_radius;
	n2_over_n1 = reference_bssrdf_rel_ior;
	albedo = reference_bssrdf_material_params->albedo.x;
	extinction = reference_bssrdf_material_params->extinction.x;
	g = reference_bssrdf_material_params->meancosine.x;
	optix_print("a %f, e %f, g %f\n", albedo, extinction, g);

	optix::float3 xi, wi, ni, xo, no;
	get_reference_scene_geometry(theta_i, r, theta_s, xi, wi, ni, xo, no);

	PhotonSample p = photon_buffer[idx];

	if (p.status == PHOTON_STATUS_NEW)
	{
		optix_print("New photon.\n");

        if(ref_frame_number == 0) {
            init_seed(p.t, ((unsigned long long) ref_frame_number) * launch_dim.x + idx);
        }
		RND_FUNC(p.t);
        if(idx == 0)
            printf("%llu %f\n", p.t.l, RND_FUNC(p.t));
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
	 
	if (scatter_photon(reference_bssrdf_output_shape, p.xp, p.wp, p.flux, reference_resulting_flux_intermediate, xo, n2_over_n1, albedo, extinction, g, p.t, p.i, batch_iterations))
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
    photon_buffer[idx].t = p.t;

}

RT_PROGRAM void reference_bssrdf_gpu_post()
{
	unsigned long long photons = 0;
	for (int i = 0; i < photon_counter.buf.size(); i++)
	{
        //optix_print("Photons %d --> %d %d\n", i, photon_counter[i], photons);
		photons += photon_counter[i];
	}
    optix_print("Photons %llu (%f)\n", photons, reference_resulting_flux_intermediate[launch_index]);
	reference_resulting_flux[launch_index] = reference_resulting_flux_intermediate[launch_index] / photons;
}



