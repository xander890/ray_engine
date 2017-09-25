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



__device__ __forceinline__ bool trace_depth_ray(const optix::float3& origin, const optix::float3& direction, float & depth, optix::float3 & normal, const float t_max = RT_DEFAULT_MAX)
{
	PerRayData_normal_depth attribute_fetch_ray_payload = { make_float3(0.0f), RT_DEFAULT_MAX };
	optix::Ray attribute_fetch_ray;
	attribute_fetch_ray.ray_type = RAY_TYPE_ATTRIBUTE;
	attribute_fetch_ray.tmin = scene_epsilon;
	attribute_fetch_ray_payload.depth = t_max;
	attribute_fetch_ray.tmax = t_max;
	attribute_fetch_ray.direction = direction;
	attribute_fetch_ray.origin = origin;

	rtTrace(current_geometry_node, attribute_fetch_ray, attribute_fetch_ray_payload);

	//optix_print("Miss? %s\n", abs(attribute_fetch_ray_payload.depth - t_max) < 1e-3 ? "true" : "false");

	if (abs(attribute_fetch_ray_payload.depth - t_max) < 1e-3) // Miss
		return false;
	depth = attribute_fetch_ray_payload.depth;
	normal = attribute_fetch_ray_payload.normal;
	return true;
}


__device__ __forceinline__ bool sample_xi_ni_from_tangent_hemisphere(const float3 & disc_origin, const float3 & disc_normal, const float3 & disc_tangent, const float3 & disc_bitangent, optix::float2 & disc_sample, uint & t, float3 & xi, float3 & ni, const float normal_bias = 0.0f)
{
	float3 sample_ray_origin = disc_origin + disc_tangent*disc_sample.x + disc_bitangent*disc_sample.y;
	float3 sample_ray_dir = disc_normal;
	sample_ray_origin += normal_bias * disc_normal; // Offsetting the ray origin along the normal. Useful to shoot rays "backwards" towards the surface
	float t_max = RT_DEFAULT_MAX;

	float depth;
	if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, depth, ni, t_max))
		return false;
	xi = sample_ray_origin + depth * sample_ray_dir;
	return true;
}

__device__ __forceinline__ int choose_sampling_axis(uint & t)
{
	float var = rnd(t);
	int main_axis = 0;
	float* mis_weights_cdf = reinterpret_cast<float*>(&bssrdf_sampling_properties->mis_weights_cdf);

	for (int i = 0; i < 4; i++)
	{
		main_axis = var > mis_weights_cdf[i] ? i : main_axis;
	}
	return main_axis;
}

__device__ __forceinline__ bool camera_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 disc_sample = sample_disk_exponential(t, chosen_sampling_mfp, pdf_disk, r, phi);

	float t_max = RT_DEFAULT_MAX;
	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;
	float3 sample_on_tangent_plane = xo + to*disc_sample.x + bo*disc_sample.y;
	float3 sample_ray_dir = normalize(sample_on_tangent_plane - camera_data.eye);
	float3 sample_ray_origin = camera_data.eye;

	float depth;
	if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, depth, ni, t_max))
		return false;
	xi = sample_ray_origin + depth * sample_ray_dir;

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
__device__ __forceinline__ bool tangent_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 disc_sample = sample_disk_exponential(t, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;

	if (!sample_xi_ni_from_tangent_hemisphere(xo, -no, to, bo, disc_sample, t, xi, ni, -bssrdf_sampling_properties->d_max))
		return false;

	integration_factor *= r / pdf_disk;
	optix_print("r: %f, pdf_disk %f, inte %f\n", r, pdf_disk, integration_factor);
	float inv_jac = max(bssrdf_sampling_properties->dot_no_ni_min, dot(normalize(no), normalize(ni)));
	integration_factor = inv_jac > 0.0f ? integration_factor / inv_jac : 0.0f;
	return true;
}			


__device__ __forceinline__ bool tangent_no_offset(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 disc_sample = sample_disk_exponential(t, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;

	const float pdf_top = 0.1f;

#define TOP 0
#define BOTTOM 1
	float verse_mult[2] = { 1, -1 };
	float verse_offset[2] = { 0.0f, -scene_epsilon * 2.0f };
	int verse = rnd(t) < pdf_top ? TOP : BOTTOM;
	float3 used_no = no*verse_mult[verse];
	integration_factor /= 1.0f-pdf_top; // ...and for the chosen axis

	const float offset = verse_offset[verse]; //including plane self intersections.
											  // We sample the main axis
	if (!sample_xi_ni_from_tangent_hemisphere(xo, used_no, to, bo, disc_sample, t, xi, ni, offset))
		return false;
#undef TOP
#undef BOTTOM


	integration_factor *= r / pdf_disk;
	optix_print("r: %f, pdf_disk %f, inte %f\n", r, pdf_disk, integration_factor);
	float inv_jac = max(bssrdf_sampling_properties->dot_no_ni_min, dot(normalize(no), normalize(ni)));
	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor = inv_jac > 0.0f ? integration_factor / inv_jac : 0.0f;
	return true;
}

__device__ __forceinline__ bool tangent_mis_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 disc_sample = sample_disk_exponential(t, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;

	optix::float3 axes[3] = { no, bo, to };
	int main_axis = choose_sampling_axis(t);
	float* mis_weights_pdf = reinterpret_cast<float*>(&bssrdf_sampling_properties->mis_weights);

	optix_print("Selecting axis %d", main_axis);

	float3 top = axes[main_axis];
	float3 t1 = axes[(main_axis + 1) % 3];
	float3 t2 = axes[(main_axis + 2) % 3];
	integration_factor /= mis_weights_pdf[main_axis];

	if (!sample_xi_ni_from_tangent_hemisphere(xo, top, t1, t2, disc_sample, t, xi, ni, -bssrdf_sampling_properties->d_max))
		return false;

	integration_factor *= r / pdf_disk;
	optix_print("r: %f, pdf_disk %f, inte %f\n", r, pdf_disk, integration_factor);
	float inv_jac = max(0.0f, dot(normalize(top), normalize(ni)));
	
	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor /= inv_jac;

	return true;
}

__device__ __forceinline__ bool tangent_mis_king(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 disc_sample = sample_disk_exponential(t, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;

	// We first choose one of the sampling axes...
	optix::float3 axes[3] = { no, bo, to };
	int main_axis = choose_sampling_axis(t);
	float* mis_weights_pdf = reinterpret_cast<float*>(&bssrdf_sampling_properties->mis_weights);
	float3 top = axes[main_axis];
	float3 t1 = axes[(main_axis + 1) % 3];
	float3 t2 = axes[(main_axis + 2) % 3];
	const float pdf_top = 0.5f;

#define TOP 0
#define BOTTOM 1
	float verse_mult[2] = { 1, -1 };
	float verse_offset[2] = { 0.0f, -scene_epsilon * 2.0f };
	int verse = rnd(t) < pdf_top ? TOP : BOTTOM;
	top *= verse_mult[verse];
	integration_factor /= mis_weights_pdf[main_axis]; // We already divide by the chosen pdf.
	integration_factor /= 1.0f - pdf_top; // ...and for the chosen axis

	const float offset = verse_offset[verse]; //including plane self intersections.
	// We sample the main axis
	if (!sample_xi_ni_from_tangent_hemisphere(xo, top, t1, t2, disc_sample, t, xi, ni, offset))
		return false;
#undef TOP
#undef BOTTOM

	integration_factor *= r / pdf_disk;
	float inv_jac = max(0.0f, dot(normalize(top), normalize(ni)));

	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor = inv_jac > 0.0f ? integration_factor / inv_jac : 0.0f;
	return true;
}

__device__ __forceinline__ bool importance_sample_position(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	switch (bssrdf_sampling_properties->sampling_method)
	{
	case BSSRDF_SAMPLING_CAMERA_BASED:				return camera_based_sampling(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BSSRDF_SAMPLING_TANGENT_PLANE:				return tangent_based_sampling(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BSSRDF_SAMPLING_MIS_AXIS:					return tangent_mis_based_sampling(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BSSRDF_SAMPLING_TANGENT_PLANE_TWO_PROBES:	return tangent_no_offset(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BSSRDF_SAMPLING_MIS_AXIS_AND_PROBES:		return tangent_mis_king(xo, no, wo, props, t, xi, ni, integration_factor);	break;
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

	float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 xo = ray.origin + t_hit*ray.direction;
	float3 wo = -ray.direction;
	float3 no = faceforward(n, wo, n);
	const MaterialDataCommon & material = get_material(xo);
	const ScatteringMaterialProperties& props = material.scattering_properties;
	float recip_ior = 1.0f / material.relative_ior;
	uint& t = prd_radiance.seed;
	float reflect_xi = rnd(t);
	prd_radiance.result = make_float3(0.0f);

#ifdef TRANSMIT
	float3 beam_T = make_float3(1.0f);
	float cos_theta_o = dot(wo, n);
	bool inside = cos_theta_o < 0.0f;
	if (inside)
	{
		beam_T = get_beam_transmittance(t_hit, props);
		float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
		if (rnd(t) >= prob) return;
		beam_T /= prob;
		recip_ior = material.relative_ior;
		cos_theta_o = -cos_theta_o;
	}

	float sin_theta_t_sqr = recip_ior*recip_ior*(1.0f - cos_theta_o*cos_theta_o);
	float cos_theta_t = 1.0f;
	float R = 1.0f;
	if (sin_theta_t_sqr < 1.0f)
	{
		cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
		R = fresnel_R(cos_theta_o, cos_theta_t, recip_ior);
	}

	R = bssrdf_sampling_properties->show_mode == BSSRDF_SHADERS_SHOW_REFLECTION ? 1.0f : R;
	R = bssrdf_sampling_properties->show_mode == BSSRDF_SHADERS_SHOW_REFRACTION ? 0.0f : R;

	if (reflect_xi >= R)
	{
		float3 wt = recip_ior*(cos_theta_o*no - wo) - no*cos_theta_t;
		PerRayData_radiance prd_refracted = prepare_new_pt_payload(prd_radiance);

		Ray refracted(xo, wt, RAY_TYPE_RADIANCE, scene_epsilon);
		rtTrace(top_object, refracted, prd_refracted);

		prd_radiance.seed = prd_refracted.seed;
		prd_radiance.result += prd_refracted.result*beam_T;

		if (!inside)
		{
#else
	float cos_theta_o = dot(wo, no);
	float R = fresnel_R(cos_theta_o, recip_ior);
#endif

	float3 L_d = make_float3(0.0f);
	uint N = samples_per_pixel;// sampling_output_buffer.size();

	int count = 0;

	for (uint i = 0; i < N; i++)
	{
		float integration_factor;
		float3 xi, ni;
		if (!importance_sample_position(xo, no, wo, props, t, xi, ni, integration_factor))
			continue;
		// Real hit point
		
		optix::float3 wi = make_float3(0);
		optix::float3 L_i;
		sample_light(xi, ni, 0, t, wi, L_i); // This returns pre-sampled w_i and L_i

		// compute direction of the transmitted light
		float cos_theta_i = max(dot(wi, ni), 0.0f);
		float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
		float sin_theta_t_sqr = recip_ior*recip_ior*(1.0f - cos_theta_i_sqr);
		float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
		float3 w12 = recip_ior*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;
		float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_t_i, recip_ior);

		float3 w21 = no * cos_theta_t - recip_ior * (cos_theta_o * no - wo);

		// compute contribution if sample is non-zero
		if (dot(L_i, L_i) > 0.0f)
		{
			float3 S_d = bssrdf(xi, ni, w12, xo, no, w21, props);
			L_d += L_i * S_d * T12 * integration_factor;
			optix_print("Ld %f %f %f Li %f %f %f T12 %f int %f\n", L_d.x, L_d.y, L_d.z, L_i.x, L_i.y, L_i.z, T12, integration_factor);
		}
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
		float3 wr = 2.0f*cos_theta_o*no - wo;
		PerRayData_radiance prd_reflected = prepare_new_pt_payload(prd_radiance);
		Ray reflected(xo, wr, RAY_TYPE_RADIANCE, scene_epsilon);
		rtTrace(top_object, reflected, prd_reflected);

		prd_radiance.seed = prd_reflected.seed;
		prd_radiance.result += prd_reflected.result;
	}
#endif
}

RT_PROGRAM void shade() { _shade(); }
RT_PROGRAM void shade_path_tracing() { _shade(); }