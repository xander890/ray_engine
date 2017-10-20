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
#define RND_FUNC rnd_tea

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


__device__ __forceinline__ bool sample_xi_ni_from_tangent_hemisphere(const float3 & disc_origin, const float3 & disc_normal, const float3 & disc_tangent, const float3 & disc_bitangent, optix::float2 & disc_sample, uint & t, float3 & xi, float3 & ni, const float normal_bias = 0.0f, const float t_min = scene_epsilon)
{
	float3 sample_ray_origin = disc_origin + disc_tangent*disc_sample.x + disc_bitangent*disc_sample.y;
	float3 sample_ray_dir = disc_normal;
	sample_ray_origin += normal_bias * disc_normal; // Offsetting the ray origin along the normal. Useful to shoot rays "backwards" towards the surface
	float t_max = RT_DEFAULT_MAX;
	if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, xi, ni, t_min, t_max))
		return false;

	return true;
}

__device__ __forceinline__ int sample_inverse_cdf(float * cdf, int cdf_size, float xi)
{
	int c = 0;
	for (int i = 0; i < cdf_size; i++)
	{
		c = xi > cdf[i] ? i : c;
	}
	return c;
}

__device__ __forceinline__ int choose_sampling_axis(uint & t)
{
	float var = RND_FUNC(t);
	float* mis_weights_cdf = reinterpret_cast<float*>(&bssrdf_sampling_properties->mis_weights_cdf);
	return sample_inverse_cdf(mis_weights_cdf, 4, var);
}

__device__ __forceinline__ bool camera_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

	float t_max = RT_DEFAULT_MAX;
	optix::float3 to, bo;
	create_onb(no, to, bo);
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
__device__ __forceinline__ bool tangent_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;

	if (!sample_xi_ni_from_tangent_hemisphere(xo, -no, to, bo, disc_sample, t, xi, ni, -bssrdf_sampling_properties->d_max))
		return false;

	integration_factor *= r / pdf_disk;
	optix_print("r: %f, pdf_disk %f, inte %f\n", r, pdf_disk, integration_factor);
	float inv_jac = max(bssrdf_sampling_properties->dot_no_ni_min, dot(normalize(no), normalize(ni)));

	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor = inv_jac > 0.0f ? integration_factor / inv_jac : 0.0f;
	return true;
}			


__device__ __forceinline__ bool tangent_no_offset(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	rnd_tea(t);
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);
	integration_factor = 1.0f;

	const float pdf = 0.5f;

#define TOP 0
#define BOTTOM 1
	float verse_mult[2] = { 1, -1 };
	float pdfs[2] = { pdf, 1 - pdf };
	int verse = rnd_tea(t) < pdf ? TOP : BOTTOM;
	float3 used_no = no * verse_mult[verse];
	integration_factor /= pdfs[verse]; // ...and for the chosen axis

	if (!sample_xi_ni_from_tangent_hemisphere(xo + no * scene_epsilon * 2, used_no, to, bo, disc_sample, t, xi, ni, 0.0f, 0.0f))
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

__device__ __forceinline__ bool random_axis(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);

	optix::float3 axes[3] = { no, bo, to };

	int main_axis = RND_FUNC(t) * 3.0f;  
	float inv_pdf_axis = 3.0f;

	float verse_mult[2] = { 1, -1 };
	int v = RND_FUNC(t) < 0.5f? 0 : 1;

	float3 probe_direction = verse_mult[v] * axes[main_axis];
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

__device__ __forceinline__ bool axis_mis_probes(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	float chosen_sampling_mfp = get_sampling_mfp(props);
	float r, phi, pdf_disk;
	optix::float2 sample = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
	optix::float2 disc_sample = sample_disk_exponential(sample, chosen_sampling_mfp, pdf_disk, r, phi);

	optix::float3 to, bo;
	create_onb(no, to, bo);

	optix::float3 axes[3] = { no, bo, to };

	// We first choose an axis.
	int main_axis = int(RND_FUNC(t) * 3.0f);
	float pdf_axis = 1.0f/3.0f;
	//integration_factor /= pdf_axis;
	float3 disc_axis = axes[main_axis];

	// Sampling the corresponding tangent plane.
	float3 disc_t, disc_b;
	create_onb(disc_axis, disc_t, disc_b);
	float3 tangent_plane_xi = xo + disc_t*disc_sample.x + disc_b*disc_sample.y;
	
	optix_print("Sampling tangent plane %d\n", main_axis);

	optix::float3 xo_xitan = xo - tangent_plane_xi;

	int axis_1 = (main_axis + 1) % 3;
	float r_axis_1 = optix::length(xo_xitan - dot(xo_xitan, axes[axis_1]) * axes[axis_1]);
	float pdf_axis_1 = exponential_pdf_disk(r_axis_1, chosen_sampling_mfp);

	int axis_2 = (main_axis + 2) % 3;
	float r_axis_2 = optix::length(xo_xitan - dot(xo_xitan, axes[axis_2]) * axes[axis_2]);
	float pdf_axis_2 = exponential_pdf_disk(r_axis_2, chosen_sampling_mfp);

	float mis_weight = pdf_disk / (pdf_disk + pdf_axis_1 + pdf_axis_2);
	
	// Probe rays.
	float pdf[6] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
	float3 xis[6];
	float3 nis[6];
	float3 all_dirs[6];

	tangent_plane_xi += disc_axis * scene_epsilon * 2; // Just to avoid problems on planar surfaces.
	float verses[2] = { 1, -1 };
	float norm = 0.0f;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			float3 probe_direction = verses[j] * axes[i];
			int idx = i * 2 + j;
			all_dirs[idx] = probe_direction;

			if (!trace_depth_ray(tangent_plane_xi, probe_direction, xis[idx], nis[idx], 0, RT_DEFAULT_MAX))
			{
				pdf[idx] = 0.0f;
				continue;  
			}
			pdf[idx] = 1.0f; // max(0.0f, dot(nis[idx], probe_direction));
			norm += pdf[idx];
		}
	}

	if (norm == 0.0f)
		return false;

	// Calculating cdf from the pdf.
	float cdf[7] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f, 0.0f };
	for (int i = 1; i <= 6; i++)
	{
		pdf[i - 1] /= norm;
		cdf[i] = cdf[i - 1] + pdf[i - 1];
	} 

	// Sampling the inverse cdf.
	float var = RND_FUNC(t);
	int axis = sample_inverse_cdf(&cdf[0], 7, var);
	float3 ax = all_dirs[axis];

	optix_print("Pdf    %f %f %f %f %f %f \n", pdf[0], pdf[1], pdf[2], pdf[3], pdf[4], pdf[5]);
	optix_print("Cdf %f %f %f %f %f %f %f\n", cdf[0], cdf[1], cdf[2], cdf[3], cdf[4], cdf[5], cdf[6]);
	optix_print("Chosen axis %d, (%f %f %f), pdf %f\n", axis, ax.x, ax.y, ax.z, var);

	ni = nis[axis];
	xi = xis[axis];

	integration_factor = 1;
	// MIS weight
	integration_factor *= mis_weight;
	// Pdf of the point (1/nk)
	integration_factor /= pdf[axis];
	// Disk pdf (cancels out partially with MIS weight)
	integration_factor *= r / pdf_disk;
	// Jacobian
	if (bssrdf_sampling_properties->use_jacobian == 1)
		integration_factor *= abs(dot(ax, ni));
	return true;
}

__device__ __forceinline__ bool importance_sample_position(const float3 & xo, const float3 & no, const float3 & wo, const ScatteringMaterialProperties& props, uint & t,
	float3 & xi, float3 & ni, float & integration_factor)
{
	switch (bssrdf_sampling_properties->sampling_method)
	{
	case BssrdfSamplingType::BSSRDF_SAMPLING_CAMERA_BASED:				return camera_based_sampling(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE:				return tangent_based_sampling(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE_TWO_PROBES:	return tangent_no_offset(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_MIS_AXIS:					return random_axis(xo, no, wo, props, t, xi, ni, integration_factor);	break;
	case BssrdfSamplingType::BSSRDF_SAMPLING_MIS_AXIS_AND_PROBES:		return axis_mis_probes(xo, no, wo, props, t, xi, ni, integration_factor);	break;
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
	float reflect_xi = RND_FUNC(t);
	prd_radiance.result = make_float3(0.0f);

#ifdef TRANSMIT
	float3 beam_T = make_float3(1.0f);
	float cos_theta_o = dot(wo, n);
	bool inside = cos_theta_o < 0.0f;
	if (inside)
	{
		beam_T = get_beam_transmittance(t_hit, props);
		float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
		if (RND_FUNC(t) >= prob) return;
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

		Ray refracted(xo, wt,  RayType::RADIANCE, scene_epsilon);
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
		{
			optix_print("Sample non valid.\n");
			continue;
		}
		// Real hit point
		
#ifdef TEST_SAMPLING
		L_d += make_float3(integration_factor * TEST_SAMPLING_W);
#else
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
		float3 wr = 2.0f*cos_theta_o*no - wo;
		PerRayData_radiance prd_reflected = prepare_new_pt_payload(prd_radiance);
		Ray reflected(xo, wr,  RayType::RADIANCE, scene_epsilon);
		rtTrace(top_object, reflected, prd_reflected);

		prd_radiance.seed = prd_reflected.seed;
		prd_radiance.result += prd_reflected.result;
	}
#endif
}

RT_PROGRAM void shade() { _shade(); }
RT_PROGRAM void shade_path_tracing() { _shade(); }