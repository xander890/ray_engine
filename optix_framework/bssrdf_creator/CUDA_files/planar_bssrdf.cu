
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

rtDeclareVariable(float, reference_bssrdf_theta_o, , ) = 0.0f;
rtDeclareVariable(float, reference_bssrdf_theta_i, , );
rtDeclareVariable(float, reference_bssrdf_theta_s, , );
rtDeclareVariable(float, reference_bssrdf_radius, , );
rtDeclareVariable(BufPtr<ScatteringMaterialProperties>, planar_bssrdf_material_params, , );
rtDeclareVariable(float, reference_bssrdf_rel_ior, , );
rtDeclareVariable(int, reference_bssrdf_output_shape, , );
//#define USE_HARDCODED_MATERIALS
 
RT_PROGRAM void reference_bssrdf_camera()
{
	float2 uv = make_float2(launch_index) / make_float2(launch_dim);

	float theta_i = reference_bssrdf_theta_i;
	float n2_over_n1 = reference_bssrdf_rel_ior;
	float albedo = planar_bssrdf_material_params->albedo.x;
	float extinction = planar_bssrdf_material_params->extinction.x;
	float g = planar_bssrdf_material_params->meancosine.x;
	float theta_s = reference_bssrdf_theta_s;
	float r = reference_bssrdf_radius;

    BSSRDFGeometry geometry;
	get_reference_scene_geometry(theta_i, r, theta_s, geometry.xi, geometry.wi, geometry.ni, geometry.xo, geometry.no);
	  
	  
	if (reference_bssrdf_output_shape == BSSRDF_OUTPUT_HEMISPHERE) 
	{   
		float2 angles = get_normalized_hemisphere_buffer_angles(uv.y, uv.x);
		geometry.wo = optix::make_float3(sinf(angles.y) * cosf(angles.x), sinf(angles.y) * sinf(angles.x), cosf(angles.y));
	}
	else  
	{      
		geometry.wo = normalize(optix::make_float3(sinf(reference_bssrdf_theta_o), 0, cosf(reference_bssrdf_theta_o)));
		optix::float2 plan = get_planar_buffer_coordinates(uv);
		geometry.xo = make_float3(plan.x, plan.y, 0);
	} 

	const float n1_over_n2 = 1.0f / n2_over_n1;
	float R12;
	optix::float3 w12; 
	refract(geometry.wi, geometry.ni, n1_over_n2, w12, R12); 
	float T12 = 1.0f - R12; 
	 
	float R21;
	optix::float3 w21;
	refract(geometry.wo, geometry.no, n1_over_n2, w21, R21); 
	float T21 = 1.0f - R21;  
	w21 = -w21;

    MaterialDataCommon mat;
    mat.scattering_properties = planar_bssrdf_material_params[0];
	optix::float3 S = T12 * bssrdf(geometry, n1_over_n2, mat) * T21;
	planar_resulting_flux_intermediate[launch_index] = S.x;
} 
      
RT_PROGRAM void post_process_bssrdf() 
{
	planar_resulting_flux[launch_index] = planar_resulting_flux_intermediate[launch_index];
}
