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

//rtDeclareVariable(unsigned int, bssrdf_enabled, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.attenuation = 0.0f;
	rtTerminateRay();
}

// Closest hit program for Lambertian shading using the basic light as a directional source
RT_PROGRAM void shade()
{
	if (prd_radiance.depth > max_depth)
	{
		prd_radiance.result = make_float3(0.0f);
		return;
	}

	optix_print("Depth %d/%d\n", prd_radiance.depth, max_depth);

	float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 xo = ray.origin + t_hit*ray.direction;
	float3 wo = -ray.direction;
	float3 no = faceforward(n, wo, n);
	const MaterialDataCommon & material = get_material(xo);
	const ScatteringMaterialProperties& props = material.scattering_properties;
	float recip_ior = 1.0f / props.relative_ior;
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
		recip_ior = props.relative_ior;
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
	if (reflect_xi >= R)
	{
		float3 wt = recip_ior*(cos_theta_o*no - wo) - no*cos_theta_t;
		PerRayData_radiance prd_refracted;
		prd_refracted.depth = prd_radiance.depth + 1;
		Ray refracted(xo, wt, RAY_TYPE_RADIANCE, scene_epsilon);
		rtTrace(top_object, refracted, prd_refracted);
		prd_radiance.result += prd_refracted.result*beam_T;

		if (!inside)
		{
#else
	float cos_theta_o = dot(wo, no);
	float R = fresnel_R(cos_theta_o, recip_ior);
#endif

	float chosen_transport_rr = props.min_transport;
	float3 L_d = make_float3(0.0f);
	uint N = 1;// sampling_output_buffer.size();

	PerRayData_normal_depth attribute_fetch_ray_payload = { make_float3(0.0f), RT_DEFAULT_MAX };
	optix::Ray attribute_fetch_ray;
	attribute_fetch_ray.ray_type = RAY_TYPE_ATTRIBUTE;
	attribute_fetch_ray.tmin = scene_epsilon;
	attribute_fetch_ray.tmax = RT_DEFAULT_MAX;
	attribute_fetch_ray.origin = camera_data.eye;
	int count = 0;

	for (uint i = 0; i < N; i++)
	{
		optix::float2 sample = make_float2(rnd(t), rnd(t));
		float r, phi;
		optix::float2 disc_sample = sample_disk_exponential(sample, chosen_transport_rr, r, phi);
		optix::float3 to, bo;
		create_onb(no, to, bo);
		optix::float3 sample_on_tangent_plane = xo + to*disc_sample.x + bo*disc_sample.y;
		attribute_fetch_ray.direction = normalize(sample_on_tangent_plane - camera_data.eye);
		attribute_fetch_ray.origin = camera_data.eye; // sample_on_tangent_plane + no * 1.0f;

		rtTrace(top_object, attribute_fetch_ray, attribute_fetch_ray_payload);

		if (attribute_fetch_ray_payload.depth == RT_DEFAULT_MAX) // Miss
			continue;
		// Real hit point
		optix::float3 xi = attribute_fetch_ray.origin + attribute_fetch_ray_payload.depth * attribute_fetch_ray.direction;
		optix::float3 ni = attribute_fetch_ray_payload.normal;
		optix::float3 wi = make_float3(0);
		optix::float3 L_i;
		sample_environment(wi, L_i, HitInfo(xi, ni), t); // This returns pre-sampled w_i and L_i

		// compute direction of the transmitted light
		float cos_theta_i = max(dot(wi, ni), 0.0f);
		float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
		float sin_theta_t_sqr = recip_ior*recip_ior*(1.0f - cos_theta_i_sqr);
		float cos_theta_t = sqrt(1.0f - sin_theta_t_sqr);
		float3 w12 = recip_ior*(cos_theta_i*ni - wi) - ni*cos_theta_t;
		float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_t, recip_ior);

		// compute contribution if sample is non-zero
		if (dot(L_i, L_i) > 0.0f)
		{
			float3 d = camera_data.eye - xi;
			float3 d_prime = camera_data.eye - sample_on_tangent_plane;
			float jacobian = cos_theta_o / cos_theta_i * dot(d_prime, d_prime) / dot(d, d);
			float3 S_d = bssrdf(xi, ni, w12, xo, no, props);
			float dist = length(xo - xi);
			float pdf_disk = chosen_transport_rr * exp(-dist * chosen_transport_rr) / (2.0f* M_PIf);
			L_d += L_i * S_d * T12 * jacobian * r / pdf_disk;
			count++;
			optix_print("sampled distance %f bssrdf %f %f %f jacobian %f Ld %f %f %f Li %f %f %f pdf\n", dist, S_d.x, S_d.y, S_d.z, jacobian, L_d.x, L_d.y, L_d.z, L_i.x, L_i.y, L_i.z, pdf_disk );
		}
	}
	count = max(1, count);
#ifdef TRANSMIT
	prd_radiance.result += L_d / (float)count;
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
		PerRayData_radiance prd_reflected;
		prd_reflected.depth = prd_radiance.depth + 1;
		Ray reflected(xo, wr, RAY_TYPE_RADIANCE, scene_epsilon);
		rtTrace(top_object, reflected, prd_reflected);
		prd_radiance.result += prd_reflected.result;
	}
#endif
}