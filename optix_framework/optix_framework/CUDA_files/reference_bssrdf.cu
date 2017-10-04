// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"
#include "device_environment_map.h"
#include "optical_helper.h"
#include "phase_function.h"

using namespace optix;
// Window variables
rtBuffer<float,2> resulting_flux;
#define MAX_ITERATIONS 1e5
//#define REMOVE_SINGLE_SCATTERING

__forceinline__ __device__ bool intersect_plane(const optix::float3 & plane_origin, const optix::float3 & plane_normal, const optix::Ray & ray, float & intersection_distance)
{
	float denom = optix::dot(plane_normal, ray.direction);
	if (abs(denom) < 1e-6)
		return false; // Parallel: none or all points of the line lie in the plane.
	intersection_distance = optix::dot((plane_origin - ray.origin), plane_normal) / denom;
	return intersection_distance > ray.tmin && intersection_distance < ray.tmax;
}

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

	// Optical properties
	const float critical_angle = asinf(n1_over_n2);
	const float n2_over_n1 = 1.0f / n1_over_n2;

	// Refraction
	const float cos_theta_i = optix::max(optix::dot(wi, ni), 0.0f);
	const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
	const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
	const float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
	const optix::float3 w12 = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;
	float flux_t = incident_power;
	optix::float3 xp = xi;
	optix::float3 wp = w12;
	int i;
	for(i = 0; i < MAX_ITERATIONS; i++)
	{
		float rand = 1.0f - rnd(t); // Avoids zero thus infinity.
		float d = -log(rand) / extinction;
		float intersection_distance;
		optix::Ray ray = optix::make_Ray(xp, wp, 0, scene_epsilon, d);
		optix_assert(xp.z < 1e-6); 
		optix_assert(xp.z > -INFINITY);

		if (!intersect_plane(xi, ni, ray, intersection_distance))
		{
			// Still within the medium...
			xp = xp + wp * d;
			optix::float3 d_vec = xo - xp;
			const float d_vec_len = optix::length(d_vec);
			d_vec = d_vec / d_vec_len; // Normalizing

			optix_assert(optix::dot(d_vec, no) > 0.0f);

			const float cos_theta_21 = optix::max(optix::dot(d_vec, no), 0.0f); // This should be positive
			const float cos_theta_21_sqr = cos_theta_21*cos_theta_21;

			// Note: we flip the relative ior because we are going outside from inside
			const float sin_theta_o_sqr = n2_over_n1*n2_over_n1*(1.0f - cos_theta_21_sqr);
			optix_print("(%d) - pos %f %f dir %f %f\n", i, xp.x, xp.y, d_vec.x, d_vec.y);

			if (sin_theta_o_sqr >= 1.0f) // Total internal reflection, no accumulation
			{
				continue;
			}
			else
			{
				float cos_theta_o = sqrt(1.0f - sin_theta_o_sqr);
				const float F_t = 1.0f - fresnel_R(cos_theta_21, n2_over_n1); // assert F_t < 1
				float phi_21 = atan2f(d_vec.y, d_vec.x);

				optix_assert(F_t < 1.0f);

				// Store atomically in appropriate spot.

				float flux_E = flux_t * albedo * eval_HG(optix::dot(d_vec, wp), g) * expf(-extinction*d_vec_len) * F_t;
				
				float phi_o = phi_21; // 
				float phi_o_normalized = normalize_angle(phi_o) / (2.0f * M_PIf);
				const float theta_o_normalized = acosf(cos_theta_o) / (M_PIf * 0.5f);


				optix_assert(theta_o_normalized >= 0.0f && theta_o_normalized < 1.0f);
				optix_assert(phi_o_normalized < 1.0f);
				optix_assert(phi_o_normalized >= 0.0f);

				float2 coords = make_float2(phi_o_normalized, theta_o_normalized);
				uint2 idxs = make_uint2(coords * make_float2(bins));
#ifdef REMOVE_SINGLE_SCATTERING
				if (i > 0)
					atomicAdd(&resulting_flux[idxs], flux_E);
#else
				atomicAdd(&resulting_flux[idxs], flux_E);
#endif



				optix_assert(flux_E >= 0.0f);
				optix_print("Scattering. (%d %d) %f\n", idxs.x, idxs.y, flux_E);
			}

			float absorption_prob = rnd(t);
			if (absorption_prob > albedo)
			{
				optix_print("(%d) Absorption.\n",i);
				break;
			}
			// We scatter, so we choose a new direction sampling the phase function
			wp = sample_HG(wp, g, t);
		}
		else
		{
			// We are going out!
			const optix::float3 surface_point = xp + wp * intersection_distance;
			optix_assert(optix::dot(wp, no) > 0);
			const float cos_theta_p = optix::max(optix::dot(wp, no), 0.0f); 
			const float cos_theta_p_sqr = cos_theta_p*cos_theta_p;

			const float sin_theta_o_sqr = n2_over_n1*n2_over_n1*(1.0f - cos_theta_p_sqr);

			if (sin_theta_o_sqr >= 1.0f) // Total internal reflection,
			{
				xp = surface_point;
				wp = reflect(wp, -no); // Reflect and turn to face inside.
				const float cos_theta_o = sqrt(1.0f - sin_theta_o_sqr);
				const float F_t = 1.0f - fresnel_R(cos_theta_o, n1_over_n2); // assert F_t < 1
				flux_t *= F_t;
				optix_print("(%d) Interally refracting.\n", i);
			}
			else
			{
				optix_print("(%d) Going out of the medium.\n", i);
				// We are outside
				break;
			}
		}
	}
	if (i == MAX_ITERATIONS)
		optix_print("Max iterations reached. Distance %f (%f times ||xo-xi||)\n", length(xp - xo), length(xp - xo)/ length(xi - xo));

}

#define MILK 0
#define WAX 1
#define A 2
#define B 3
#define C 4
#define D 5
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
#endif
	const float theta_i_rad = deg2rad(theta_i);
	const float theta_s_rad = deg2rad(-theta_s);
	const optix::float3 wi = normalize(optix::make_float3(-sinf(theta_i_rad), 0, cosf(theta_i_rad)));
	infinite_plane_scatter_searchlight(wi, incident_power, r, theta_s_rad, n1_over_n2, albedo, extinction, g, t);
}

