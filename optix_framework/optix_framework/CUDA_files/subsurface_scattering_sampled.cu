// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#define TRANSMIT
#define REFLECT 

#include <device_common_data.h>
#include <math_helpers.h>
#include <random.h>
#include <bssrdf.h>
#include <optical_helper.h>
#include <structs.h>
#include <ray_trace_helpers.h>
#include <scattering_properties.h>
#include <material_device.h>
#include <light.h>
#include <sampling_helpers.h>
#include <camera.h>
#include <device_environment_map.h>
#include <neural_network_device_code.h>

using namespace optix;

//#define REFLECT

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// SS properties

rtDeclareVariable(CameraData, camera_data, , );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(unsigned int, samples_per_pixel, , );

rtDeclareVariable(BufPtr<BSSRDFSamplingProperties>, bssrdf_sampling_properties, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.attenuation = 0.0f;
	rtTerminateRay();
}



__device__ __forceinline__ bool trace_depth_ray(const optix::float3& origin, const optix::float3& direction, optix::float3 & xi, optix::float3 & normal, const float t_min = scene_epsilon, const float t_max = RT_DEFAULT_MAX)
{
	PerRayData_normal_depth attribute_fetch_ray_payload = { make_float3(0.0f), RT_DEFAULT_MAX };
	optix::Ray attribute_fetch_ray;
	attribute_fetch_ray.ray_type =  RayType::ATTRIBUTE;
	attribute_fetch_ray.tmin = t_min;
	attribute_fetch_ray_payload.depth = t_max;
	attribute_fetch_ray.tmax = t_max;
	attribute_fetch_ray.direction = direction;
	attribute_fetch_ray.origin = origin;

	rtTrace(current_geometry_node, attribute_fetch_ray, attribute_fetch_ray_payload);

	optix_print("Miss? %s, dir %f %f %f\n", abs(attribute_fetch_ray_payload.depth - t_max) < 1e-3 ? "true" : "false", direction.x, direction.y, direction.z);

	if (abs(attribute_fetch_ray_payload.depth - t_max) < 1e-9f) // Miss
		return false;
	xi = origin + attribute_fetch_ray_payload.depth * direction;
	normal = attribute_fetch_ray_payload.normal;
	return true;
}


__device__ __forceinline__ void sample_point_on_normal_tangent_plane(
        const float3 & xo,          // The points hit by the camera ray.
        const float3 & no,          // The normal at the point.
        const float3 & wo,          // The incoming ray direction.
        const MaterialDataCommon & material,  // Material properties.
        TEASampler * sampler,       // A rng.
	    float3 & x_tangent,                // The candidate point 
        float & integration_factor, // An factor that will be multiplied into the final result. For inverse pdfs. 
        bool & has_candidate_wi,    // Returns true if the point has a candidate outgoing direction
        float3 & proposed_wi)       // The candidate proposed direction.
{
	switch (bssrdf_sampling_properties->sampling_tangent_plane_technique)
	{
        case BssrdfSamplePointOnTangentTechnique::EXPONENTIAL_DISK:
        {
            optix::float3 to, bo;
            create_onb(no, to, bo);
	        const ScatteringMaterialProperties& props = material.scattering_properties;
	        float chosen_sampling_mfp = get_sampling_mfp(props);
	        float r, phi, pdf_disk;

	        optix::float2 sample = optix::make_float2(sampler->next1D(), sampler->next1D());
	        optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);
	        integration_factor *= r / pdf_disk;
            x_tangent = xo + r * cosf(phi) * to + r * sinf(phi) * bo;
            has_candidate_wi = false;

        } break;
        case BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING:
        {
            has_candidate_wi = true;
            sample_neural_network(xo,no,wo,material, sampler, x_tangent, integration_factor, proposed_wi);        
        } break;
    }
}


__device__ __forceinline__ bool sample_xi_ni_from_tangent_hemisphere(const float3 & disc_point, const float3 & disc_normal, float3 & xi, float3 & ni, const float normal_bias = 0.0f, const float t_min = scene_epsilon)
{
	float3 sample_ray_origin = disc_point;
	float3 sample_ray_dir = disc_normal;
	sample_ray_origin += normal_bias * disc_normal; // Offsetting the ray origin along the normal. to shoot rays "backwards" towards the surface
	float t_max = RT_DEFAULT_MAX;
	if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, xi, ni, t_min, t_max))
		return false;

	return true;
}

__device__ __forceinline__ bool camera_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, TEASampler * sampler,
	float3 & xi, float3 & ni, float & integration_factor)
{

	const ScatteringMaterialProperties& props = material.scattering_properties;
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(sampler->next1D(), sampler->next1D());
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

    optix::float3 to, bo;
    create_onb(no, to, bo);
	float t_max = RT_DEFAULT_MAX;
	integration_factor = 1.0f;
	float3 sample_on_tangent_plane = xo + to*disc_sample.x + bo*disc_sample.y;
	float3 sample_ray_dir = normalize(sample_on_tangent_plane - camera_data.eye);
	float3 sample_ray_origin = camera_data.eye;

	if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, xi, ni, scene_epsilon, t_max))
		return false;

	integration_factor *= r / pdf_disk;
	optix_print("r: %f, pdf_disk %f, inte %f\n", r, pdf_disk, integration_factor);

	if (bssrdf_sampling_properties->use_jacobian == 1)
	{
		float3 d = camera_data.eye - xi;
		float cos_alpha = dot(-sample_ray_dir, ni);

		float3 d_tan = camera_data.eye - sample_on_tangent_plane;
		float cos_alpha_tan = dot(-sample_ray_dir, no);

		float jacobian = max(1e-3, cos_alpha_tan) / max(1e-3, cos_alpha) * max(1e-3, dot(d, d)) / max(1e-3, dot(d_tan, d_tan));
		integration_factor *= jacobian;
	}

	return true;
}
__device__ __forceinline__ bool tangent_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, TEASampler * sampler,
	float3 & xi, float3 & ni, float & integration_factor, bool & has_candidate_wi, float3 & proposed_wi)
{

    optix::float3 xo_tangent;
    sample_point_on_normal_tangent_plane(xo,no,wo,material,sampler, xo_tangent, integration_factor, has_candidate_wi, proposed_wi);

	if (!sample_xi_ni_from_tangent_hemisphere(xo_tangent, -no, xi, ni, -bssrdf_sampling_properties->d_max))
		return false;

	float inv_jac = max(bssrdf_sampling_properties->dot_no_ni_min, dot(normalize(no), normalize(ni)));

	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor *= inv_jac > 0.0f ? 1 / inv_jac : 0.0f;
	return true;
}			


__device__ __forceinline__ bool tangent_no_offset(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, TEASampler * sampler,
	float3 & xi, float3 & ni, float & integration_factor, bool & has_candidate_wi, float3 & proposed_wi)
{
    optix::float3 xo_tangent;
    sample_point_on_normal_tangent_plane(xo,no,wo,material,sampler, xo_tangent, integration_factor, has_candidate_wi, proposed_wi);

	const float pdf = 0.5f;

#define TOP 0
#define BOTTOM 1
	float verse_mult[2] = { 1, -1 };
	float pdfs[2] = { pdf, 1 - pdf };
	int verse = sampler->next1D() < pdf ? TOP : BOTTOM;
	float3 used_no = no * verse_mult[verse];
	integration_factor /= pdfs[verse]; // ...and for the chosen axis

	if (!sample_xi_ni_from_tangent_hemisphere(xo_tangent, used_no, xi, ni, 0.0f, 0.0f))
		return false;
#undef TOP
#undef BOTTOM

	float inv_jac = max(bssrdf_sampling_properties->dot_no_ni_min, dot(normalize(no), normalize(ni)));

	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor = inv_jac > 0.0f ? integration_factor / inv_jac : 0.0f;
	return true;
}

__device__ __forceinline__ bool axis_mis_probes(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, TEASampler * sampler,
	float3 & xi, float3 & ni, float & integration_factor)
{
	const ScatteringMaterialProperties& props = material.scattering_properties;
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(sampler->next1D(), sampler->next1D());
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);

	optix::float3 axes[3] = { no, bo, to };

	int main_axis = sampler->next1D() * 3.0f;  
	float inv_pdf_axis = 3.0f;

	float verse = sampler->next1D() < 0.5f? -1 : 1;

	float3 probe_direction = verse * axes[main_axis];
	float inv_pdf = 2.0f * inv_pdf_axis;

	optix::float3 chosen_axes[3] = { probe_direction, make_float3(0), make_float3(0) };
	create_onb(probe_direction, chosen_axes[1], chosen_axes[2]);

	float3 xi_tangent_space = xo + chosen_axes[1]*disc_sample.x + chosen_axes[2]*disc_sample.y;
	float3 sample_ray_dir = probe_direction;
	float t_max = RT_DEFAULT_MAX;

	if (!trace_depth_ray(xi_tangent_space + no * scene_epsilon * 2, sample_ray_dir, xi, ni, 0.0f, t_max))
		return false;

	optix::float3 xo_xi = xo - xi;

	float dot0 = abs(dot(ni, probe_direction));

	float3 axis_1 = chosen_axes[1];
	float dot1 = abs(dot(ni, axis_1));
	float3 axis_2 = chosen_axes[2];
	float dot2 = abs(dot(ni, axis_2));

	float wi0 = pdf_disk * dot0 / r;

	float3 tangent_point_xi_xo_1 = xo_xi - dot(xo_xi, axis_1) * axis_1;
	float r_axis_1 = optix::length(tangent_point_xi_xo_1);
	float pdf_axis_1 = exponential_pdf_disk(r_axis_1, chosen_sampling_mfp);
	float J_1_inv = abs(dot(ni, axis_1));
	float wi1 = pdf_axis_1 * dot1 / r_axis_1;

	float3 tangent_point_xi_xo_2 = xo_xi - dot(xo_xi, axis_2) * axis_2;
	float r_axis_2 = optix::length(tangent_point_xi_xo_2);
	float pdf_axis_2 = exponential_pdf_disk(r_axis_2, chosen_sampling_mfp);
	float J_2_inv = abs(dot(ni, axis_2));
	float wi2 = pdf_axis_2 * dot2 / r_axis_2;

	float weight = 1.0f / (wi0 + wi1 + wi2);

	integration_factor = inv_pdf * weight;
	return true;
}

__device__ __forceinline__ bool importance_sample_position(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, TEASampler * sampler,
	float3 & xi, float3 & ni, float & integration_factor, bool & has_candidate_wi, float3 & proposed_wi)
{
    has_candidate_wi = false;
	switch (bssrdf_sampling_properties->sampling_method)
	{
	case BssrdfSamplingType::BSSRDF_SAMPLING_CAMERA_BASED:				return camera_based_sampling(xo, no, wo, material, sampler, xi, ni, integration_factor);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE:				return tangent_based_sampling(xo, no, wo, material, sampler, xi, ni, integration_factor, has_candidate_wi, proposed_wi);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE_TWO_PROBES:	return tangent_no_offset(xo, no, wo, material, sampler, xi, ni, integration_factor, has_candidate_wi, proposed_wi);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_MIS_AXIS:					return axis_mis_probes(xo, no, wo, material, sampler, xi, ni, integration_factor);	break;
	}
}


// Closest hit program for Lambertian shading using the basic light as a directional source
__device__ __forceinline__ void _shade()
{
	if (prd_radiance.depth > max_depth)
	{
		prd_radiance.result = make_float3(0.0f);
		return;
	}

	TEASampler * sampler = prd_radiance.sampler;
	float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 xo = ray.origin + t_hit*ray.direction;
	float3 wo = -ray.direction;
	float3 no = faceforward(n, wo, n);
	const MaterialDataCommon material = get_material(xo);
	const ScatteringMaterialProperties& props = material.scattering_properties;
	float recip_ior = 1.0f / material.relative_ior;
	float reflect_xi = sampler->next1D();
	prd_radiance.result = make_float3(0.0f);

#ifdef TRANSMIT
	float3 beam_T = make_float3(1.0f);
	float cos_theta_o = dot(wo, n);
	bool inside = cos_theta_o < 0.0f;
	if (inside)
	{
		beam_T = get_beam_transmittance(t_hit, props);
		float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
		if (sampler->next1D() >= prob) return;
		beam_T /= prob;
		recip_ior = material.relative_ior;
		cos_theta_o = -cos_theta_o;
	}

	float3 wt;
	float R;
	refract(wo, n, recip_ior, wt, R);

	R = bssrdf_sampling_properties->show_mode == BSSRDF_SHADERS_SHOW_REFLECTION ? 1.0f : R;
	R = bssrdf_sampling_properties->show_mode == BSSRDF_SHADERS_SHOW_REFRACTION ? 0.0f : R;

	if (reflect_xi >= R)
	{
		PerRayData_radiance prd_refracted = prepare_new_pt_payload(prd_radiance);

		Ray refracted(xo, wt,  RayType::RADIANCE, scene_epsilon);
		rtTrace(top_object, refracted, prd_refracted);

		prd_radiance.result += prd_refracted.result*beam_T;

		if (!inside)
		{
#else
	float cos_theta_o = dot(wo, no);
	float R = fresnel_R(cos_theta_o, recip_ior);
#endif

	float3 L_d = make_float3(0.0f);
	uint N = samples_per_pixel;
	int count = 0;

	for (uint i = 0; i < N; i++)
	{
		float integration_factor = 1.0f;
		float3 xi, ni;
        bool has_candidate_wi;
        float3 proposed_wi;

		if (!importance_sample_position(xo, no, wo, material, sampler, xi, ni, integration_factor, has_candidate_wi, proposed_wi))
		{
			optix_print("Sample non valid.\n");
			continue;
		}
		
#ifdef TEST_SAMPLING
		L_d += make_float3(integration_factor * TEST_SAMPLING_W);
#else
		optix::float3 wi = make_float3(0);
		optix::float3 L_i;
		sample_light(xi, ni, 0, sampler, wi, L_i); // This returns pre-sampled w_i and L_i

		// compute direction of the transmitted light
		
        float3 w12; 
	    float R12;
	    refract(wi, ni, recip_ior, w12, R12);
		float T12 = 1.0f - R12;

		// compute contribution if sample is non-zero
		if (dot(L_i, L_i) > 0.0f)
		{
            float3 S;
            if(bssrdf_sampling_properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING)
            {
                 S = make_float3(1);
            }
            else
            {
                 BSSRDFGeometry geometry;
                 geometry.xi = xi;
                 geometry.ni = ni;
                 geometry.wi = wi;
                 geometry.xo = xo;
                 geometry.no = no;
                 geometry.wo = wo;
                 S = bssrdf(geometry, recip_ior, material);
            }

			// INCLUDE FALSE
			L_d += L_i * S * integration_factor;
			optix_print("Sd %e %e %e Ld %f %f %f Li %f %f %f T12 %f int %f\n",  S.x, S.y, S.z, L_d.x, L_d.y, L_d.z, L_i.x, L_i.y, L_i.z, T12, integration_factor);
		}
#endif
	}
#ifdef TRANSMIT
		prd_radiance.result += L_d / (float)N;
		}
	}
#else
	float T21 = 1.0f - R;
	prd_radiance.result += T21*accumulate / (float)count;
#endif
#ifdef REFLECT
	// Trace reflected ray
	if (reflect_xi < R)
	{
		
		float3 wr = -reflect(wo, no);
		PerRayData_radiance prd_reflected = prepare_new_pt_payload(prd_radiance);
		Ray reflected(xo, wr,  RayType::RADIANCE, scene_epsilon);
		rtTrace(top_object, reflected, prd_reflected);

		prd_radiance.result += prd_reflected.result;
	}
#endif
}

RT_PROGRAM void shade() { _shade(); }
RT_PROGRAM void shade_path_tracing() { _shade(); }
